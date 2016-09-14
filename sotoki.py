#!/usr/bin/env python2
"""sotoki.

Usage:
  sotoki.py run <work> <url> <publisher>
  sotoki.py load <work>
  sotoki.py render questions <work> <title> <publisher>
  sotoki.py render users <work> <title> <publisher>
  sotoki.py render tags <work> <title> <publisher>
  sotoki.py benchmark xml <work>
  sotoki.py benchmark wiredtiger <work>
  sotoki.py (-h | --help)
  sotoki.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""
import os
import shlex
from itertools import chain
from time import sleep
from time import time
from collections import OrderedDict
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

from lxml.etree import iterparse
from lxml.etree import parse as string2xml
from lxml.html import fromstring as string2html
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
from setproctitle import setproctitle
from subprocess32 import check_output
from subprocess32 import TimeoutExpired

setproctitle('sotoki')


DEBUG = os.environ.get('DEBUG')


# multiprocessing helper

class Worker(Process):

    def __init__(self, queue, function):
        super(Worker, self).__init__()
        self.queue = queue
        self.function = function

    def run(self):
        function = self.function
        for args in iter(self.queue.get, None):
            try:
                function(args)
            except Exception as exc:
                print('{} failed with {}'.format(function, exc))

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


class Boolean(Integer):
    format = 'q'

    def from_raw(self, value):
        if isinstance(value, str):
            return False if value == 'False' else True
        else:
            return False if value == 0 else True

    def to_raw(self, value):
        return 1 if value else 0


class String(Property):
    format = 'S'

    def from_raw(self, value):
        try:
            return value.decode('utf-8')
        except:
            return value

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
            setattr(self, property.name, property.from_raw(value))

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
        return cls.__name__ + 's.xml'

    @classmethod
    def table(cls):
        return 'table:' + cls.__name__

    @classmethod
    def indices_format(cls):
        for name, columns in getattr(cls, 'indices', dict()).items():
            table = 'index:{}:{}'.format(cls.__name__, name)
            columns = 'columns=({})'.format(','.join([column.name for column in columns]))
            yield table, columns

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

    def __getitem__(self, key):  # HACK: to avoid to rework all the templates
        return getattr(self, key)

class Badge(Element):
    UserId = Integer()
    Name = String()
    Date = DateTime()
    TagBased = Boolean()
    Class = Integer()

    indices = {'UserId': [UserId]}


class Comment(Element):
    UserDisplayName = String()
    PostId = Integer()
    Score = Integer()
    Text = String()
    CreationDate = DateTime()
    UserId = Integer()

    indices = {'PostId': [PostId, CreationDate]}


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

    indices = {'PostTypeId': [PostTypeId],
               'Answers': [ParentId, Score]}

    def pseudo(self):
        return u'{} {}'.format(self.Id, self.Title)


class PostLink(Element):
    CreationDate = DateTime()
    PostId = Integer()
    RelatedPostId = Integer()
    LinkTypeId = Integer()  # 1 Link, 3 Duplicate

    indices = {'PostId': [PostId]}

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
    LastAccessDate = DateTime()
    AccountId = Integer()
    ProfileImageUrl = String()

    def pseudo(self):
        return u'{} {}'.format(self.Id, self.DisplayName)

class Tag(Element):
    Count = Integer()
    WikiPostId = Integer()
    TagName = String()
    ExcerptPostId = Integer()

    indices = {'TagName': [TagName]}

class TagPost(Element):
    TagName = String()
    PostId = Integer()

    indices = {'TagName': [TagName]}

# load

def iterate(filepath):
    with open(filepath) as f:
        tree = iterparse(f)
        for event, row in tree:
            if event == 'end' and row.tag == 'row':
                yield {key: row.get(key) for key in row.keys()}
                # XXX: don't forget to clean
                # cf. http://stackoverflow.com/a/9814580/140837
                row.clear()
                # second, delete previous siblings 
                while row.getprevious() is not None:
                    del row.getparent()[0]

def load(work):
    # prepare paths
    dump = os.path.join(work, 'dump')
    db = os.path.join(work, 'db')

    # prepare wiredtiger
    connection = wiredtiger_open(db, "create")
    session = connection.open_session(None)
    # load Post and TagPost
    filepath = os.path.join(dump, Post.filename())
    print('* loading {}'.format(filepath))
    # create post table
    session.create(Post.table(), Post.format())
    for index, columns in Post.indices_format():
        session.create(index, columns)
    # create TagPost table
    session.create(TagPost.table(), TagPost.format())
    for index, columns in TagPost.indices_format():
        session.create(index, columns)
    # process
    cursor_post = session.open_cursor(Post.table(), None, None)
    cursor_tag = session.open_cursor(TagPost.table(), None, None)
    uid = 1  # FIXME: This can be done by wiredtiger, requires to fix the ORM
    for data in iterate(filepath):
        # store Post
        post = Post(**data)
        cursor_post.set_key(*post.keys())
        cursor_post.set_value(*post.values())
        cursor_post.insert()
        # store TagPost
        try:
            tags = post.Tags[1:-1].split('><')
        except KeyError:
            pass
        else:
            for tag in tags:
                cursor_tag.set_key(uid)
                cursor_tag.set_value(tag, post.Id)
                cursor_tag.insert()
                uid += 1
    cursor_post.close()
    cursor_tag.close()

    # load others
    for klass in [Badge, Comment, PostLink, User, Tag]:
        filepath = os.path.join(dump, klass.filename())
        print('* loading {}'.format(filepath))
        session.create(klass.table(), klass.format())
        for index, columns in klass.indices_format():
            session.create(index, columns)
        cursor = session.open_cursor(klass.table(), None, '')
        for data in iterate(filepath):
            object = klass(**data)
            cursor.set_key(*object.keys())
            cursor.set_value(*object.values())
            cursor.insert()
        cursor.close()

    connection.close()


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


ENV = None  # Jinja environment singleton

def jinja(output, template, templates, **context):
    global ENV
    # XXX: make the environment a singleton otherwise it creates a memory leak
    if ENV is None:
        templates = os.path.abspath(templates)
        ENV = Environment(loader=FileSystemLoader((templates,)))
        filters = dict(
            markdown=markdown,
            intspace=intspace,
            scale=scale,
            clean=lambda y: filter(lambda x: x not in punctuation, y),
            slugify=slugify,
        )
        ENV.filters.update(filters)

    template = ENV.get_template(template)
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
            except Exception as exc:
                print "Problem with image : " + filepath
                print exc
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


def chunks(iterable, size):
    """Yield successive chunks of length size from iterable."""
    out = list()
    while True:
        end = False
        out = list()
        for i in xrange(size):
            try:
                value = next(iterable)
            except StopIteration:
                end = True
                break
            else:
                out.append(value)
        if end:
            yield out
            break
        else:
            yield out
            continue


class Context(object):

    def __init__(self, session):
        self.table_comment = session.open_cursor('table:Comment', None, None)
        self.table_tag = session.open_cursor('table:Tag', None, None)
        self.table_user = session.open_cursor('table:User', None, None)
        self.table_post = session.open_cursor('table:Post', None, None)
        self.index_comment = session.open_cursor('index:Comment:PostId(Id)', None, None)
        self.index_question = session.open_cursor('index:Post:PostTypeId(Id)', None, None)
        self.index_answer = session.open_cursor('index:Post:Answers(Id)', None, None)
        self.index_link = session.open_cursor('index:PostLink:PostId(RelatedPostId)', None, None)
        self.index_tagpost = session.open_cursor('index:TagPost:TagName(PostId)', None, None)


def db_get_comment(context, uid):
    cursor = context.table_comment
    cursor.set_key(uid)
    cursor.search()
    values = cursor.get_value()
    comment = Comment(uid, *values)
    cursor.reset()
    return comment

def db_get_user(context, uid):
    cursor = context.table_user
    cursor.set_key(uid)
    if cursor.search() == 0:
        user = User(uid, *cursor.get_value())
        cursor.reset()
        return user
    else:
        return None

def db_get_post(context, uid):
    posts = context.table_post
    posts.set_key(uid)
    if posts.search() != 0:
        return None
    post = posts.get_value()
    post = Post(uid, *post)
    # get comments
    index = context.index_comment
    index.set_key(post.Id, '')
    post.comments = list()
    if index.search() == 0:
        while True:
            comment = index.get_value()
            comment = db_get_comment(context, comment)
            post.comments.append(comment)
            if index.next() == 0:
                if index.get_key() == post.Id:
                    continue
            break
    if post.OwnerUserId:
        post.OwnerUserId = db_get_user(context, post.OwnerUserId)
    index.reset()
    posts.reset()
    return post


def render_question(args):
    build, templates, title, publisher, question = args
    filename = slugify(question.pseudo()) + '.html'
    filepath = os.path.join(build, filename)

    # offline images
    for post in chain([question], question.answers):

        try:
            body = string2html(post.Body)
        except Exception as exc:
            print 'string2html failed for post Id:{}'.format(post.Id)
            print exc
        else:
            imgs = body.xpath('//img')
            dirty = False
            for img in imgs:
                src = img.attrib['src']
                ext = os.path.splitext(src)[1]
                filename = sha1(src).hexdigest() + ext
                out = os.path.join(build, '..', 'static', 'images', filename)
                # download the image only if it's not already downloaded
                if os.path.exists(out):
                    continue
                try:
                    download(src, out)
                except Exception as exc:
                    print 'download {} failed with:'.format(src)
                    print exc
                else:
                    # update post's html
                    src = '../static/images/' + filename
                    img.attrib['src'] = src
                    # finalize offlining
                    try:
                        resize(out)
                        optimize(out)
                    except Exception as exc:
                        print "resize or optimize of {} failed".format(src)
                        print exc
                    else:
                        dirty = True
            if dirty:
                post.Body = html2string(body)

    # properly order by highest score
    question.answers = reversed(question.answers)

    jinja(
        filepath,
        'question.html',
        templates,
        question=question,
        rooturl="..",
        title=title,
        publisher=publisher,
    )


def render_questions(work, title, publisher, cores):
    print 'render questions'
    # prepare paths
    templates = os.path.abspath('templates')
    database = os.path.join(work, 'db')
    build = os.path.join(work, 'build', 'question')
    images = os.path.join(work, 'build', '..', 'static', 'images')
    dump = os.path.join(work, 'dump')
    # prepare database
    connection = wiredtiger_open(database, "create")
    session = connection.open_session(None)
    context = Context(session)

    try:
        os.makedirs(os.path.join(work, 'build', 'question'))
    except:
        pass
    try:
        os.makedirs(os.path.join(work, 'build', 'static', 'images'))
    except:
        pass
    try:
        os.makedirs(os.path.join(work, 'build', 'static', 'identicon'))
    except:
        pass

    def db_iter_questions():
        questions = context.index_question
        questions.set_key(1)
        questions.search()
        while True:
            question = questions.get_value()
            question = db_get_post(context, question)
            # format tags
            tags = question['Tags']
            tags = tags[1:-1].split('><')
            question.Tags = tags
            # get answers
            question.answers = list()
            answers = context.index_answer
            answers.set_key(question.Id, 0)
            if answers.search() == 0:
                while True:
                    answer = answers.get_value()
                    answer = db_get_post(context, answer)
                    question.answers.append(answer)
                    if answers.next() == 0:
                        if answers.get_key() == question.Id:
                            continue
                    break

            answers.reset()
            # get links
            question.links = list()
            links = context.index_link
            links.set_key(question.Id)
            if links.search() == 0:
                while True:
                    # FIXME: related and duplicates are mixed
                    link = links.get_value()
                    link = db_get_post(context, link)
                    if link:
                        question.links.append(link)
                    if links.next() == 0:
                        if links.get_key() == question.Id:
                            continue
                    break
            links.reset()

            yield build, templates, title, publisher, question

            if questions.next() == 0:
                if questions.get_key() == 1:
                    continue
            break

    queue = Queue()
    workers = list()
    for i in range(cores):
        worker = Worker(queue, render_question)
        workers.append(worker)
        worker.start()

    for i, args in enumerate(db_iter_questions()):
        queue.put(args)
        if DEBUG and i == 10:  # debug
            break

    for worker in workers:
        queue.put(None)

    for worker in workers:
        worker.join()

    connection.close()

def render_tag(args):
    fullpath, tag, page, questions, title, publisher = args
    questions.sort(key=lambda x: x.Score, reverse=True)
    jinja(
        fullpath,
        'tag.html',
        'templates',
        tag=tag,
        index=page,
        questions=questions,
        rooturl="../..",
        hasnext=True,  # FIXME
        next=page + 1,
        title=title,
        publisher=publisher,
    )

def render_tags(work, title, publisher, cores):
    # FIXME: use pool.map
    print 'render tags'
    # prepare paths
    templates = os.path.abspath('templates')
    database = os.path.join(work, 'db')
    build = os.path.join(work, 'build')
    # prepare database
    connection = wiredtiger_open(database, "create")
    session = connection.open_session(None)
    context = Context(session)

    tagpost = context.index_tagpost

    # render index page
    tags = list()
    cursor = context.table_tag
    while cursor.next() == 0:
        uid = cursor.get_key()
        values = cursor.get_value()
        tag = Tag(uid, *values)
        tags.append(tag)
    cursor.reset()

    tags.sort(key=lambda x: x.TagName)

    jinja(
        os.path.join(build, 'index.html'),
        'tags.html',
        templates,
        tags=tags,
        rooturl=".",
        title=title,
        publisher=publisher,
    )

    # render tag page using pagination
    tags = [tag.TagName for tag in tags]
    try:
        os.makedirs(os.path.join(build, 'tag'))
    except:
        pass

    def iter_tags():
        for i, tag in enumerate(tags):
            dirpath = os.path.join(build, 'tag')
            tagpath = os.path.join(dirpath, tag)
            try:
                os.makedirs(tagpath)
            except:
                pass
            # build page using pagination
            # FIXME: support true next page
            questions = list()
            tagpost.set_key(tag)
            if tagpost.search() != 0:
                continue
            while True:
                uid = tagpost.get_value()
                question = db_get_post(context, uid)
                questions.append(question)

                if tagpost.next() == 0:
                    if  tagpost.get_key() == tag:
                        continue
                break

            for page, chunk in enumerate(chunks(iter(questions), 10)):
                fullpath = os.path.join(tagpath, '%s.html' % page)
                yield fullpath, tag, page, chunk, title, publisher

    queue = Queue()
    workers = list()
    for i in range(cores):
        worker = Worker(queue, render_tag)
        workers.append(worker)
        worker.start()

    for i, args in enumerate(iter_tags()):
        queue.put(args)
        if DEBUG and i == 10:  # debug
            break

    for worker in workers:
        queue.put(None)

    for worker in workers:
        worker.join()

    connection.close()

def render_user(args):
    user, generator, build, title, publisher = args
    username = slugify(user.pseudo())

    # Generate big identicon
    padding = (20, 20, 20, 20)
    identicon = generator.generate(username, 164, 164, padding=padding, output_format="png")  # noqa
    filename = username + '.png'
    fullpath = os.path.join(build, 'static', 'identicon', filename)
    with open(fullpath, "wb") as f:
        f.write(identicon)

    # Generate small identicon
    padding = [0] * 4  # no padding
    identicon = generator.generate(username, 32, 32, padding=padding, output_format="png")  # noqa
    filename = username + '.small.png'
    fullpath = os.path.join(build, 'static', 'identicon', filename)

    with open(fullpath, "wb") as f:
        f.write(identicon)

    # generate user profile page
    filename = '%s.html' % username
    fullpath = os.path.join(build, 'user', filename)
    jinja(
        fullpath,
        'user.html',
        'templates',
        user=user,
        title=title,
        publisher=publisher,
        rooturl='..',
    )


def render_users(work, title, publisher, cores):
    print 'render users'
    templates = 'templates'
    database = os.path.join(work, 'db')
    build = os.path.join(work, 'build')
    try:
        os.makedirs(os.path.join(build, 'user'))
    except:
        pass

    # Prepare identicon generation
    identicon_path = os.path.join(build, 'static', 'identicon')
    try:
        os.makedirs(identicon_path)
    except:
        pass

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

    connection = wiredtiger_open(database, "create")
    session = connection.open_session(None)
    context = Context(session)

    def db_iter_users():
        cursor = context.table_user
        cursor.reset()
        while cursor.next() == 0:
            uid = cursor.get_key()
            values = cursor.get_value()
            user = User(uid, *values)

            yield user, generator, build, title, publisher

            if cursor.next() == 0:
                continue
            else:
                break

    queue = Queue()
    workers = list()
    for i in range(cores):
        worker = Worker(queue, render_user)
        workers.append(worker)
        worker.start()

    for i, args in enumerate(db_iter_users()):
        queue.put(args)
        if DEBUG and i == 10:  # debug
            break

    for worker in workers:
        queue.put(None)

    for worker in workers:
        worker.join()

    connection.close()


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


def exec_cmd(cmd, timeout=None):
    try:
        return check_output(shlex.split(cmd), timeout=timeout)
    except TimeoutExpired:
        pass


def create_zims(work, title, publisher, description):
    print 'Creating ZIM files'
    # Check, if the folder exists. Create it, if it doesn't.
    lang_input = "en"
    html_dir = os.path.join(work, "build")
    zim_path = dict(
        title=title.lower(),
        lang=lang_input,
        date=datetime.datetime.now().strftime('%Y-%m')
    )
    zim_path = os.path.join(work, "{title}_{lang}_all_{date}.zim".format(**zim_path))

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


if __name__ == '__main__':
    args = docopt(__doc__, version='sotoki 0.3')
    if args['run']:
        # if not bin_is_present("zimwriterfs"):
        #     sys.exit("zimwriterfs is not available, please install it.")
        cores = cpu_count() - 1 or 1
        work = args['<work>']
        url = args['<url>']
        publisher = args['<publisher>']
        # load dump into database
        load(work)
        # render templates into `output`
        templates = 'templates'
        build = os.path.join('work', 'build')
        try:
            os.makedirs(build)
        except:
            pass
        title, description = grab_title_description_favicon(url, build)
        render_questions(work, title, publisher, cores)
        render_tags(work, title, publisher, cores)
        render_users(work, title, publisher, cores)
        # copy static
        copy_tree('static', os.path.join(work, 'build', 'static'))
        # create_zims(work, title, publisher, description)
    elif args['benchmark']:
        if args['xml']:
            print('* Running benchmark for lxml')
            dump = os.path.join(args['<work>'], 'dump')

            filepath = os.path.join(dump, Post.filename())
            print('** loading {}'.format(filepath))
            for data in iterate(filepath):
                obj = Post(**data)
        elif args['wiredtiger']:
            print '* Running benchmark for wiredtiger'
            # prepare paths
            work = args['<work>']
            dump = os.path.join(work, 'dump')
            database = os.path.join(work, 'db')
            # prepare database
            connection = wiredtiger_open(database, 'create')
            session = connection.open_session(None)
            # create table and indices
            session.create(Post.table(), Post.format())
            for index, columns in Post.indices_format():
                session.create(index, columns)
            cursor = session.open_cursor('table:Post', None, None)
            print '** Loading Post.xml'
            filepath = os.path.join(dump, Post.filename())
            for data in iterate(filepath):
                obj = Post(**data)
                cursor.set_key(*obj.keys())
                cursor.set_value(*obj.values())
                cursor.insert()
            print('* Iterate over whole table')
            cursor.reset()
            while cursor.next() == 0:
                uid = cursor.get_key()
                values = cursor.get_value()
            connection.close()
    elif args['load']:
        load(args['<work>'])
    elif args['render']:
        if args['questions']:
            cores = cpu_count() - 1 or 1
            render_questions(args['<work>'], args['<title>'], args['<publisher>'], cores)
        elif args['users']:
            cores = cpu_count() - 1 or 1
            render_users(args['<work>'], args['<title>'], args['<publisher>'], cores)
        elif args['tags']:
            cores = cpu_count() - 1 or 1
            render_tags(args['<work>'], args['<title>'], args['<publisher>'], cores)
