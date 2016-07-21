#!/usr/bin/env python2
"""sotoki.

Usage:
  sotoki.py run <url> <publisher> [--directory=<dir>]
  sotoki.py load <dump-directory> <database-directory>
  sotoki.py render <templates> <database> <output> <title> <publisher> [--directory=<dir>]
  sotoki.py render-users <templates> <database> <output> <title> <publisher> [--directory=<dir>]
  sotoki.py offline <output> <cores>
  sotoki.py (-h | --help)
  sotoki.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --directory=<dir>   Specify a directory for xml files [default: work/dump/]
"""
import os
from collections import OrderedDict
import xml.etree.cElementTree as etree
import logging

import re
import os.path
from hashlib import sha1
from distutils.dir_util import copy_tree
from urllib2 import urlopen
from string import punctuation

from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import Queue
from multiprocessing import Process

from wiredtiger import wiredtiger_open

from jinja2 import Environment
from jinja2 import FileSystemLoader

from lxml.etree import parse as string2xml
from lxml.html import parse as html
from lxml.html import tostring as html2string

from PIL import Image
from resizeimage import resizeimage

from docopt import docopt
from slugify import slugify
from markdown import markdown as md
import pydenticon

import bs4 as BeautifulSoup
import envoy
import sys
import datetime
import subprocess


# wiredtiger orm

# Generic declarative API. Does not support references/fk
# forked from https://gist.github.com/amirouche/4384b5b2f02b469fb8e3

class Property(object):
	"""Base class data-descriptor for element properties.

	Subclass it to implement new types. Most likely you will
	need to implement `unwrap` and `wrap`.
	"""

	COUNTER = 0

	def __init__(self, **options):
		self.options = options
		# set by metaclass
		self.klass = None
		self.name = None
		self.uid = Property.COUNTER + 1
		Property.COUNTER += 1

	def __repr__(self):
		return '<Property class:{} name:{} uid:{}>'.format(self.klass, self.name, self.uid)

	def __get__(self, element, cls=None):
		if element is None:
			# `Property` instance is accessed through a class
			return self
		else:
			return element._values[self.name]

	def __set__(self, element, value):
		"""Set the `value` for `element`"""
		element._values[self.name] = value

	def __delete__(self, element):
		del element._values[self.name]


class No5(type):
	"""Metaclass for Element classes"""

	def __init__(klass, classname, bases, namespace):
		klass._properties = OrderedDict()

		for base in bases:
			try:
				klass._properties.update(base._properties)
			except AttributeError:
				pass  # not a valid Element

		# register properties defined for this class
		properties = [(a, b) for a, b in namespace.items() if isinstance(b, Property)]
		properties.sort(key=lambda x: x[1].uid)
		for name, attribute in properties:
			klass._properties[name] = attribute
			attribute.name = name
			attribute.klass = klass.__name__


class Integer(Property):
	format = 'q'

	def from_raw(self, value):
		return int(value)

	def to_raw(self, value):
		return value or 0


class String(Property):
	format = 's'

	def from_raw(self, value):
		return value  # FIXME:f or some reason it's already utf8

	def to_raw(self, value):
		return (value or '').encode('utf-8')


class DateTime(String):
	pass


class Element(object):

	__metaclass__ = No5

	Id = Integer(key=True)

	def __init__(self, *args, **kwargs):
		self._values = OrderedDict()
		# assign args to properties using the declaration order of properties.
		# This is possible because `self._properties` is an `OrderedDict`
		properties = self._properties.values()
		for value, property in zip(args, properties):
			setattr(self, property.name, value)

		# assign `kwargs`. Overrides already set values...
		for name, value in kwargs.items():
			try:
				property = self._properties[name]
			except KeyError:
				msg = '`{}` Element class has no `{}` property'
				msg = msg.format(type(self), name)
				raise ValueError(msg)
			else:
				setattr(self, name, property.from_raw(value))

	@classmethod
	def filename(cls):
		return cls.__name__.lower() + 's.xml'

	@classmethod
	def table(cls):
		return 'table:' + cls.__name__

	@classmethod
	def format(cls):
		properties = cls._properties.values()
		keys = [p.format for p in properties if p.options.get('key', False)]
		values = [p.format for p in properties if not p.options.get('key', False)]
		config = 'key_format=' + ''.join(keys) + ',value_format=' + ''.join(values)
		names = [p.name for p in properties]
		config += ',columns=(' + ','.join(names) + ')'
		return config

	def keys(self):
		def iter():
			for property in self._properties.values():
				if property.options.get('key', False):
					value = self._values[property.name]
					value = property.to_raw(value)
					yield value
		return list(iter())

	def values(self):
		def iter():
			for property in self._properties.values():
				if not property.options.get('key', False):
					value = self._values.get(property.name)
					value = property.to_raw(value)
					yield value
		return list(iter())


class Badge(Element):
	UserId = Integer()
	Name = String()
	Date = String()


class Comment(Element):
	UserDisplayName = String()
	PostId = Integer()
	Score = Integer()
	Text = String()
	CreationDate = DateTime()
	UserId = Integer()


class Post(Element):
	PostTypeId = Integer()  # 1 question, 2 Answer
	ParentId = Integer()
	AcceptedAnswerId = Integer()
	CreationDate = DateTime()
	Score = Integer()
	ViewCount = Integer()
	Body = String()
	OwnerUserId = Integer()
	OwnerDisplayName = String()
	LastEditorUserId = Integer()
	LastEditorDisplayName = String()
	LastEditDate = DateTime()
	LastActivityDate = DateTime()
	CommunityOwnedDate = DateTime()
	ClosedDate = DateTime()
	Title = String()
	Tags = String()
	AnswerCount = Integer()
	CommentCount = Integer()
	FavoriteCount = Integer()


class PostLink(Element):
	CreationDate = DateTime()
	PostId = Integer()
	RelatedPostId = Integer()
	PostLinkTypeId = Integer()  # 1 Link, 3 Duplicate


class User(Element):
	Reputation = Integer()
	CreationDate = String()
	DisplayName = String()
	EmailHash = String()
	LastActivityDate = DateTime()
	WebsiteUrl = String()
	Location = String()
	Age = Integer()
	AboutMe = String()
	Views = Integer()
	UpVotes = Integer()
	DownVotes = Integer()


# helper

class Worker(Process):
	def __init__(self, queue):
		super(Worker, self).__init__()
		self.queue = queue

	def run(self):
		print 'Computing things!'
		for data in iter(self.queue.get, None):
			# Use data
			some_questions(*data)

# templating

def intspace(value):
	orig = str(value)
	new = re.sub("^(-?\d+)(\d{3})", '\g<1> \g<2>', orig)
	if orig == new:
		return new
	else:
		return intspace(new)


def markdown(text):
	# FIXME: add postprocess step to transform 'http://' into a link
	# strip p tags
	return md(text)[3:-4]


def dict_factory(cursor, row):
	d = {}
	for idx, col in enumerate(cursor.description):
		d[col[0]] = row[idx]
	return d


def scale(number):
	"""Convert number to scale to be used in style to color arrows
	and comment score"""
	if number < 0:
		return 'negative'
	if number == 0:
		return 'zero'
	if number < 3:
		return 'positive'
	if number < 8:
		return 'good'
	return 'verygood'


def jinja(output, template, templates, **context):
	templates = os.path.abspath(templates)
	env = Environment(loader=FileSystemLoader((templates,)))
	filters = dict(
		markdown=markdown,
		intspace=intspace,
		scale=scale,
		clean=lambda y: filter(lambda x: x not in punctuation, y),
		slugify=slugify,
	)
	env.filters.update(filters)
	template = env.get_template(template)
	page = template.render(**context)
	with open(output, 'w') as f:
		f.write(page.encode('utf-8'))


def download(url, output):
	response = urlopen(url)
	output_content = response.read()
	with open(output, 'w') as f:
		f.write(output_content)


def resize(filepath):
	exts = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', '.gif', '.GIF')
	if os.path.splitext(filepath)[1] in exts:
		img = Image.open(filepath)
		w, h = img.size
		if w >= 540:
			# hardcoded size based on website layout
			try:
				img = resizeimage.resize_width(img, 540, Image.ANTIALIAS)
			except:
				print "Problem with image : " + filepath
		img.save(filepath, img.format)


def optimize(filepath):
	# based on mwoffliner code http://bit.ly/1HZgZeP
	ext = os.path.splitext(filepath)[1]
	if ext in ('.jpg', '.jpeg', '.JPG', '.JPEG'):
		exec_cmd('jpegoptim --strip-all -m50 "%s"' % filepath)
	elif ext in ('.png', '.PNG'):
		# run pngquant
		cmd = 'pngquant --verbose --nofs --force --ext="%s" "%s"'
		cmd = cmd % (ext, filepath)
		exec_cmd(cmd)
		# run advancecomp
		exec_cmd('advdef -q -z -4 -i 5 "%s"' % filepath)
	elif ext in ('.gif', '.GIF'):
		exec_cmd('gifsicle -O3 "%s" -o "%s"' % (filepath, filepath))
	else:
		print('* unknown file extension %s' % filepath)


def process(args):
	images, filepaths, uid = args
	count = len(filepaths)
	print 'offlining start', uid
	for index, filepath in enumerate(filepaths):
		print 'offline %s/%s (%s)' % (index, count, uid)
		try:
			body = html(filepath)
		except Exception as exc:  # error during xml parsing
			print exc
		else:
			imgs = body.xpath('//img')
			for img in imgs:
				src = img.attrib['src']
				ext = os.path.splitext(src)[1]
				filename = sha1(src).hexdigest() + ext
				out = os.path.join(images, filename)
				# download the image only if it's not already downloaded
				if not os.path.exists(out):
					try:
						download(src, out)
					except:
						# do nothing
						pass
					else:
						# update post's html
						src = '../static/images/' + filename
						img.attrib['src'] = src
						# finalize offlining
						try:
							resize(out)
							optimize(out)
						except:
							print "Something went wrong with" + out
			# does the post contain images? if so, we surely modified
			# its content so save it.
			if imgs:
				post = html2string(body)
				with open(filepath, 'w') as f:
					f.write(post)
	print 'offlining finished', uid


def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]


def offline(output, cores):
	"""offline, resize and reduce size of images"""
	print 'offline images of %s using %s process...' % (output, cores)
	images_path = os.path.join(output, 'static', 'images')
	if not os.path.exists(images_path):
		os.makedirs(images_path)

	filepaths = os.path.join(output, 'question')
	filepaths = map(lambda x: os.path.join(output, 'question', x), os.listdir(filepaths))  # noqa
	filepaths_chunks = chunks(filepaths, len(filepaths) / cores)
	filepaths_chunks = list(filepaths_chunks)

	# start offlining
	pool = Pool(cores)
	# prepare a list of (images_path, filepaths_chunck) to feed
	# `process` function via pool.map
	args = zip([images_path]*cores, filepaths_chunks, range(cores))
	print 'start offline process with', cores, 'cores'
	pool.map(process, args)


def render_questions(templates, database, output, title, publisher, dump, cores):
	# wrap the actual database
	print 'render questions'
	db = os.path.join(database, 'se-dump.db')
	conn = sqlite3.connect(db)
	conn.row_factory = dict_factory
	cursor = conn.cursor()
	# create table tags-questions
	sql = "CREATE TABLE IF NOT EXISTS questiontag(id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, Score INTEGER, Title TEXT, CreationDate TEXT, Tag TEXT)"
	cursor.execute(sql)
	conn.commit()
	os.makedirs(os.path.join(output, 'question'))
	request_queue = Queue()
	for i in range(cores):
		Worker(request_queue).start()
	offset = 0
	while offset is not None:
		questions = cursor.execute("SELECT * FROM posts WHERE PostTypeId == 1 LIMIT 1001 OFFSET ? ",  ( offset, ) ).fetchall()
		try:
			questions[1000]
		except IndexError:
			offset = None
		else:
			offset += 1000
		questions = questions[:1000]
		for question in questions:
			question["Tags"] = question["Tags"][1:-1].split('><')
			for t in question["Tags"]:
				sql = "INSERT INTO QuestionTag(Score, Title, CreationDate, Tag) VALUES(?, ?, ?, ?)"
				cursor.execute(sql, (question["Score"], question["Title"], question["CreationDate"], t))
			user = cursor.execute("SELECT DisplayName, Reputation  FROM users WHERE Id == ? ", (str(question["OwnerUserId"]),)).fetchone()
			question["OwnerUserId"] = user
			question["comments"] = cursor.execute("SELECT * FROM comments WHERE Id == ? ", (str(question["Id"]),)).fetchall()
			for u in question["comments"]:
				tmp = cursor.execute("SELECT DisplayName  FROM users WHERE Id == ?", (str(u["UserId"]),)).fetchone()
				if tmp is not None:
					u["UserDisplayName"] = tmp["DisplayName"]
			question["answers"] = cursor.execute("SELECT * FROM posts WHERE PostTypeId == 2 AND ParentID == ? ", (str(question["Id"]),)).fetchall()
			for q in question["answers"]:
				user = cursor.execute("SELECT DisplayName, Reputation  FROM users WHERE Id == ? ", (str(q["OwnerUserId"]),)).fetchone()
				q["OwnerUserId"] = user
				q["comments"] = cursor.execute("SELECT * FROM comments WHERE Id == ? ", (str(q["Id"]),)).fetchall()
				for u in q["comments"]:
					tmp = cursor.execute("SELECT DisplayName FROM users WHERE Id == ? ", (str(u["UserId"]),)).fetchone()
					if tmp is not None:
						u["UserDisplayName"] = tmp["DisplayName"]
			tmp = cursor.execute("SELECT PostId FROM postlinks WHERE RelatedPostId == ? ", (str(question["Id"]),)).fetchall()
			question["relateds"] = []
			for links in tmp:
				name = cursor.execute("SELECT Title FROM posts WHERE Id == ? ", (links["PostId"],)).fetchone()
				if name is not None:
					question["relateds"].append(name["Title"])
			data_send = [templates, database, output, title, publisher, dump, question]
			request_queue.put(data_send)
		conn.commit()
	for i in range(cores):
		request_queue.put(None)


def some_questions(templates, database, output, title, publisher, dump, question):
	filename = '%s.html' % slugify(question["Title"])
	filepath = os.path.join(output, 'question', filename)
	try:
		jinja(
			filepath,
			'question.html',
			templates,
			question=question,
			rooturl="..",
			title=title,
			publisher=publisher,
		)
	except:
		print ' * failed to generate: %s' % filename


def render_tags(templates, database, output, title, publisher, dump):
	print 'render tags'
	# index page
	db = os.path.join(database, 'se-dump.db')
	conn = sqlite3.connect(db)
	conn.row_factory = dict_factory
	cursor = conn.cursor()

	tags = cursor.execute("SELECT TagName FROM tags ORDER BY TagName").fetchall()
	jinja(
		os.path.join(output, 'index.html'),
		'tags.html',
		templates,
		tags=tags,
		rooturl=".",
		title=title,
		publisher=publisher,
	)
	# tag page
	print "Render tag page"
	list_tag = map(lambda d: d['TagName'], tags)
	os.makedirs(os.path.join(output, 'tag'))
	for tag in list(set(list_tag)):
		dirpath = os.path.join(output, 'tag')
		tagpath = os.path.join(dirpath, '%s' % tag)
		os.makedirs(tagpath)
		print tagpath
		# build page using pagination
		offset = 0
		page = 1
		while offset is not None:
			fullpath = os.path.join(tagpath, '%s.html' % page)
			questions = cursor.execute("SELECT * FROM questiontag WHERE Tag = ? LIMIT 11 OFFSET ? ", (str(tag), offset,)).fetchall()
			try:
				questions[10]
			except IndexError:
				offset = None
			else:
				offset += 10
			questions = questions[:10]
			jinja(
				fullpath,
				'tag.html',
				templates,
				tag=tag,
				index=page,
				questions=questions,
				rooturl="../..",
				hasnext=bool(offset),
				next=page + 1,
				title=title,
				publisher=publisher,
			)
			page += 1
	conn.close()


def render_users(templates, database, output, title, publisher, dump):
	print 'render users'
	os.makedirs(os.path.join(output, 'user'))
	db = os.path.join(database, 'se-dump.db')
	conn = sqlite3.connect(db)
	conn.row_factory = dict_factory
	cursor = conn.cursor()
	users = cursor.execute("""SELECT * FROM users""").fetchall()

	# Prepare identicon generation
	identicon_path = os.path.join(output, 'static', 'identicon')
	os.makedirs(identicon_path)
	# Set-up a list of foreground colours (taken from Sigil).
	foreground = [
		"rgb(45,79,255)",
		"rgb(254,180,44)",
		"rgb(226,121,234)",
		"rgb(30,179,253)",
		"rgb(232,77,65)",
		"rgb(49,203,115)",
		"rgb(141,69,170)"
	]
	# Set-up a background colour (taken from Sigil).
	background = "rgb(224,224,224)"

	# Instantiate a generator that will create 5x5 block identicons
	# using SHA1 digest.
	generator = pydenticon.Generator(5, 5, foreground=foreground, background=background)  # noqa

	for user in users:
		username = slugify(user["DisplayName"])

		# Generate big identicon
		padding = (20, 20, 20, 20)
		identicon = generator.generate(username, 164, 164, padding=padding, output_format="png")  # noqa
		filename = username + '.png'
		fullpath = os.path.join(output, 'static', 'identicon', filename)
		with open(fullpath, "wb") as f:
			f.write(identicon)

		# Generate small identicon
		padding = [0] * 4  # no padding
		identicon = generator.generate(username, 32, 32, padding=padding, output_format="png")  # noqa
		filename = username + '.small.png'
		fullpath = os.path.join(output, 'static', 'identicon', filename)
		with open(fullpath, "wb") as f:
			f.write(identicon)

		# generate user profile page
		filename = '%s.html' % username
		fullpath = os.path.join(output, 'user', filename)
		jinja(
			fullpath,
			'user.html',
			templates,
			user=user,
			title=title,
			publisher=publisher,
		)


def grab_title_description_favicon(url, output_dir):
	output = urlopen(url).read()
	soup = BeautifulSoup.BeautifulSoup(output)
	title = soup.find('meta', attrs={"name": u"twitter:title"})['content']
	description = soup.find('meta', attrs={"name": u"twitter:description"})['content']
	favicon = soup.find('link', attrs={"rel": u"image_src"})['href']
	if favicon[:2] == "//":
		favicon = "http:" + favicon
	favicon_out = os.path.join(output_dir, 'favicon.png')
	download(favicon, favicon_out)
	resize_image_profile(favicon_out)
	return [title, description]


def resize_image_profile(image_path):
	image = Image.open(image_path)
	w, h = image.size
	image = image.resize((48, 48), Image.ANTIALIAS)
	image.save(image_path)


def exec_cmd(cmd):
	return envoy.run(str(cmd.encode('utf-8'))).status_code


def create_zims(title, publisher, description):
	print 'Creating ZIM files'
	# Check, if the folder exists. Create it, if it doesn't.
	lang_input = "en"
	html_dir = os.path.join("work", "output")
	zim_path = dict(
		title=title.lower(),
		lang=lang_input,
		date=datetime.datetime.now().strftime('%Y-%m')
	)
#    zim_path = "work/", "{title}_{lang}_all_{date}.zim".format(**zim_path)
	zim_path = os.path.join("work/", "{title}_{lang}_all_{date}.zim".format(**zim_path))

	title = title.replace("-", " ")
	creator = title
	create_zim(html_dir, zim_path, title, description, lang_input, publisher, creator)


def create_zim(static_folder, zim_path, title, description, lang_input, publisher, creator):
	print "\tWritting ZIM for {}".format(title)
	context = {
		'languages': lang_input,
		'title': title,
		'description': description,
		'creator': creator,
		'publisher': publisher,
		'home': 'index.html',
		'favicon': 'favicon.png',
		'static': static_folder,
		'zim': zim_path
	}

	cmd = ('zimwriterfs --welcome="{home}" --favicon="{favicon}" '
		   '--language="{languages}" --title="{title}" '
		   '--description="{description}" '
		   '--creator="{creator}" --publisher="{publisher}" "{static}" "{zim}"'
		   .format(**context))
	print cmd

	if exec_cmd(cmd) == 0:
		print "Successfuly created ZIM file at {}".format(zim_path)
	else:
		print "Unable to create ZIM file :("


def bin_is_present(binary):
	try:
		subprocess.Popen(binary,
						 universal_newlines=True,
						 shell=False,
						 stdin=subprocess.PIPE,
						 stdout=subprocess.PIPE,
						 stderr=subprocess.PIPE,
						 bufsize=0)
	except OSError:
		return False
	else:
		return True

def iterate(dump, filename):
	with open(os.path.join(dump, filename)) as f:
		tree = etree.iterparse(f)
		for event, row in tree:
			if event == 'end' and row.tag == 'row':
				yield {key: row.get(key) for key in row.keys()}

def load(dump, database):
	connection = wiredtiger_open(database, "create")
	session = connection.open_session(None)

	for klass in [Post, PostLink, User]:
		print('* loading {}'.format(klass.filename()))
		session.create(klass.table(), klass.format())
		cursor = session.open_cursor(klass.table(), None, '')
		for data in iterate(dump, klass.filename()):
			object = klass(**data)
			cursor.set_key(*object.keys())
			cursor.set_value(*object.values())
			cursor.insert()
		cursor.close()

	connection.close()


if __name__ == '__main__':
	arguments = docopt(__doc__, version='sotoki 0.1')
	if arguments['load']:
		load(arguments['<dump-directory>'], arguments['<database-directory>'])
	elif arguments['render']:
		render_questions(
			arguments['<templates>'],
			arguments['<database>'],
			arguments['<output>'],
			arguments['<title>'],
			arguments['<publisher>'],
			arguments['--directory']
		)
		render_tags(
			arguments['<templates>'],
			arguments['<database>'],
			arguments['<output>'],
			arguments['<title>'],
			arguments['<publisher>'],
			arguments['--directory']
		)

	elif arguments['render-users']:
		render_users(arguments['<templates>'], arguments['<database>'], arguments['<output>'])  # noqa
	elif arguments['offline']:
		offline(arguments['<output>'], int(arguments['<cores>']))
	elif arguments['run']:
		if not bin_is_present("zimwriterfs"):
			sys.exit("zimwriterfs is not available, please install it.")
		# load dump into database
		url = arguments['<url>']
		publisher = arguments['<publisher>']
		dump = arguments['--directory']
		database = 'work'
		load(dump, database)
		# render templates into `output`
		templates = 'templates'
		output = os.path.join('work', 'output')
		os.makedirs(output)
		cores = cpu_count() / 2 or 1
		title, description = grab_title_description_favicon(url, output)
		render_questions(templates, database, output, title, publisher, dump, cores)
		render_tags(templates, database, output, title, publisher, dump)
		render_users(templates, database, output, title, publisher, dump)
		# offline images
		offline(output, cores)
		# copy static
		copy_tree('static', os.path.join('work', 'output', 'static'))
		create_zims(title, publisher, description)
