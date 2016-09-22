#!/usr/bin/env python2
# -*-coding:utf8 -*
"""sotoki.

Usage:
  sotoki.py run <url> <publisher> [--directory=<dir>]
  sotoki.py (-h | --help)
  sotoki.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --directory=<dir>   Specify a directory for xml files [default: work/dump/]
"""
import time
import shutil
import uuid
import redis
import sqlite3
import os
import xml.etree.cElementTree as etree
import logging
from itertools import chain
from subprocess32 import check_output
from subprocess32 import TimeoutExpired
import shlex

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

from jinja2 import Environment
from jinja2 import FileSystemLoader

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


class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        for data in iter(self.queue.get, None):
            try:
                data[0](*data[1:])
                #some_questions(*data)
            except Exception as exc:
                print 'error while rendering question:', data[-1]['Id']
                print exc
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

def jinja(output, template, templates, raw, **context):
    global ENV
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
    if raw:
        page = "{% raw %}" + page + "{% endraw %}"
    with open(output, 'w') as f:
        f.write(page.encode('utf-8'))

def iterate(filepath):
    items = string2xml(filepath).getroot()
    for index, item in enumerate(items.iterchildren()):
        yield item.attrib

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


def comments(templates, output_tmp, dump_path, cores, uuid):
    print "Load and render comments"
    os.makedirs(os.path.join(output_tmp, 'comments'))
    r = redis.Redis('localhost')
    request_queue = Queue()
    workers = []
    for i in range(cores):
        workers.append(Worker(request_queue))
        
    for i in workers:
        i.start()

    with open(os.path.join(dump_path, "comments.xml")) as xml_file:
        tree = etree.iterparse(xml_file)
        for events, row in tree:
            try:
                comment = dict_to_unicodedict(dict(zip(row.attrib.keys(), row.attrib.values()))) 
                if comment != {}:
                    if comment.has_key("UserId"):
                        comment["UserDisplayName"] = dict_to_unicodedict(r.hgetall(uuid + "user" + str(comment["UserId"])))["DisplayName"]
                    if not comment.has_key("UserId") and not comment.has_key("UserDisplayName"):
                        comment["UserDisplayName"] = u"None"
                    filename = '%s.html' % comment["Id"]
                    filepath = os.path.join(output_tmp, 'comments', filename)
                    data_send = [some_comments, output_tmp, comment, filepath]
                    request_queue.put(data_send)
                    #some_comments(output_tmp,comment)
                    r.rpush(uuid + "post" + str(comment["PostId"]) + "comments" , os.path.join("tmp" , "comments" , filename ))
            except Exception, e:
                    print 'fail in a comments'
                    print e
    for i in range(cores):
        request_queue.put(None)
    for i in workers: 
        i.join()


def post_type2(templates, output_tmp, dump_path, cores, uuid):
    print "First passage to posts.xml"
    os.makedirs(os.path.join(output_tmp, 'post'))
    r = redis.Redis('localhost')
    request_queue = Queue()
    workers = []
    for i in range(cores):
        workers.append(Worker(request_queue))
        
    for i in workers:
        i.start()


    with open(os.path.join(dump_path, "posts.xml")) as xml_file:
        tree = etree.iterparse(xml_file)
        for events, row in tree:
            try:
                post = dict_to_unicodedict(dict(zip(row.attrib.keys(), row.attrib.values()))) 
                if post != {} and int(post["PostTypeId"]) == 2:
                    if post.has_key("OwnerUserId"):
                        post["OwnerUserId"] =   dict_to_unicodedict(r.hgetall( uuid + "user" + str(post["OwnerUserId"])))
                    elif  post.has_key("OwnerDisplayName"):
                        post["OwnerUserId"] = { "DisplayName" : post["OwnerDisplayName"].decode('utf8') }
                    else:
                        post["OwnerUserId"] = { "DisplayName" : u"None" }
                    commentaires = r.lrange(uuid + "post" + str(post["Id"]) + "comments", 0, -1 )
                    if commentaires != []:
                            post["comments"] = commentaires 
                
                    filename = '%s.html' % post["Id"]
                    filepath = os.path.join(output_tmp, 'post', filename)
                    data_send = [some_post2, output_tmp,post, filepath, output ]
                    request_queue.put(data_send)
                    r.rpush(uuid + "post" + str(post["ParentId"]) + "post2" , os.path.join("tmp" , "post" , filename ))
                elif post != {} and int(post["PostTypeId"]) == 1:
                    r.set(uuid + "post" + str(post["Id"]) + "title", post["Title"])
            except Exception, e:
                    print 'fail in a post2' + str(e)
                    print post
    for i in range(cores):
        request_queue.put(None)
    for i in workers: 
        i.join()


def render_questions(templates, database, output, title, publisher, dump, cores, uuid):
    r = redis.Redis('localhost') 
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
    workers = []
    for i in range(cores):
        workers.append(Worker(request_queue))
        
    for i in workers:
        i.start()

    with open(os.path.join(dump, "posts.xml")) as xml_file:
        tree = etree.iterparse(xml_file)
        for events, row in tree:
            try:
                question = dict_to_unicodedict(dict(zip(row.attrib.keys(), row.attrib.values()))) 
                if question != {} and int(question["PostTypeId"]) == 1:
                    question["Tags"] = question["Tags"][1:-1].split('><')
                    for t in question["Tags"]:
                        sql = "INSERT INTO QuestionTag(Score, Title, CreationDate, Tag) VALUES(?, ?, ?, ?)"
                        cursor.execute(sql, (question["Score"], question["Title"], question["CreationDate"], t))
                    if question.has_key("OwnerUserId"):
                        question["OwnerUserId"] = dict_to_unicodedict(r.hgetall(uuid + "user" + str(question["OwnerUserId"])))
                    elif question.has_key("OwnerDisplayName"):
                        question["OwnerUserId"] = { "DisplayName" : question["OwnerDisplayName"].decode('utf8') }
                    else:
                        question["OwnerUserId"] =  { "DisplayName" : u"None" }
                    question["comments"] = r.lrange(uuid + "post" + str(question["Id"]) + "comments", 0, -1 ) 
                    question["answers"] =  r.lrange(uuid + "post" + str(question["Id"]) + "post2", 0, -1 ) 
                    tmp =  r.lrange(uuid + "post" + str(question["Id"]) + "link", 0, -1 ) 
                    question["relateds"] = []
                    for link in tmp:
                        name = r.get(uuid + "post" + link + "title")
                        if name is not None:
                            question["relateds"].append(name.decode('utf8'))
                    data_send = [some_questions , templates, database, output, title, publisher, dump, question, "question.html"]
                    request_queue.put(data_send)
                    #some_questions(templates, database, output, title, publisher, dump, question, "question.html" )
                conn.commit()
            except Exception, e:
                print "error with post type 1" + str(e)
                print question
    for i in range(cores):
        request_queue.put(None)
    for i in workers: 
        i.join()


def posts_links(templates, output_tmp, dump_path, uuid):
    print "Load links"
    r = redis.Redis('localhost')
    with open(os.path.join(dump_path, "postlinks.xml")) as xml_file:
        tree = etree.iterparse(xml_file)
        for events, row in tree:
            try:
                link = dict_to_unicodedict(dict(zip(row.attrib.keys(), row.attrib.values())))
                if link != {}:
                    r.rpush(uuid + "post" + str(link["RelatedPostId"]) + "link" , link["PostId"])
            except Exception, e:
                print "error with link" + str(e)

def image(post, output):
    images = os.path.join(output, 'static', 'images') 
    body = string2html(post['Body'])
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
            except Exception,e:
                # do nothing
                print e
                pass
            else:
                # update post's html
                src = '../static/images/' + filename
                img.attrib['src'] = src
                # finalize offlining
                try:
                    resize(out)
                except:
                    print "Something went wrong with" + out
    # does the post contain images? if so, we surely modified
    # its content so save it.
    if imgs:
        body = html2string(body)
        post['Body'] = body
    return post

def some_questions(templates, database, output, title, publisher, dump, question, template_name):
    filename = '%s.html' % slugify(question["Title"])
    filepath = os.path.join(output, 'question', filename)
    question = image(question,output)
    try:
        jinja(
            filepath,
            template_name,
            templates,
            False,
            question=question,
            rooturl="..",
            title=title,
            publisher=publisher,
        )
    except Exception, e:
        print ' * failed to generate: %s' % filename
        print "erreur jinja" + str(e)
        print question

def some_comments(output_tmp,comment, filepath):
            try:
                jinja(
                    filepath,
                    'comment.html',
                    templates,
                    True,
                    comment=comment,
                )
            except Exception as exc:
                print 'error while rendering comment:', comment['Id']
                print exc


def some_post2(output_tmp,post, filepath, output):
    post = image(post, output)
    try:
        jinja(
            filepath,
            'post.mixin.html',
            templates,
            True,
            post=post,
        )
    except Exception as exc:
                print 'error while rendering post answers:', post['Id']
                print exc


def render_tags(templates, database, output, title, publisher, dump):
    print 'Render tags'
    
    # index page
    db = os.path.join(database, 'se-dump.db')
    conn = sqlite3.connect(db)
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    tags = []
    with open(os.path.join(dump, "tags.xml")) as xml_file:
        tree = etree.iterparse(xml_file)
        for events, row in tree:
            try:
                tag = dict(zip(row.attrib.keys(), row.attrib.values()))
                if tag != {}:
                    tags.append({'TagName': tag["TagName"]})
            except Exception,e:
                print "error on tag" + str(e)
                print tag
    jinja(
        os.path.join(output, 'index.html'),
        'tags.html',
        templates,
        False,
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
                False,
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



def grab_title_description_favicon(url, output_dir):
    output = urlopen(url).read()
    soup = BeautifulSoup.BeautifulSoup(output, 'html.parser')
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
        check_output(shlex.split(cmd), timeout=timeout)
    except:
        pass

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

def dict_to_unicodedict(dictionnary):
    dict_ = {}
    if dictionnary.has_key("OwnerDisplayName"):
        dictionnary["OwnerDisplayName"] = u""
    for k, v in dictionnary.items():
        unicode_key = k.decode('utf8')
        if isinstance(v, unicode):
            unicode_value = v
        else:
            unicode_value =  v.decode('utf8')
        dict_[unicode_key] = unicode_value
        
    return dict_ 

def del_redis_keys(uuid):
    r = redis.Redis('localhost')
    for key in r.scan_iter(uuid + "*"):
        r.delete(key)


def load_user(dump_path, templates, database, output, title, publisher, uuid):
    print "Load and render users"
    r = redis.Redis('localhost')    
    identicon_path = os.path.join(output, 'static', 'identicon')
    os.makedirs(identicon_path)
    os.makedirs(os.path.join(output, 'user'))
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


    with open(os.path.join(dump_path, "users.xml")) as xml_file:
        tree = etree.iterparse(xml_file)
        for events, row in tree:
            try:
                user = dict_to_unicodedict(dict(zip(row.attrib.keys(), row.attrib.values())))  
                if user != {}:
                    r.hset(uuid + "user" + user["Id"], "DisplayName", user["DisplayName"])
                    r.hset(uuid + "user" + user["Id"], "Reputation", user["Reputation"])
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
                        False,
                        user=user,
                        title=title,
                        publisher=publisher,
                    )
            except Exception, e:
                print e

def optimize(output):
    print "optimize images"
    print "jpegoptim --strip-all -m50 " + output + "/*.{jpg,jpeg}"
    exec_cmd("jpegoptim --strip-all -m50 " + output + "/*.{jpg,jpeg}", timeout=None)
    print "pngquant --verbose --nofs --force --ext=.png " + output + "/*.png"
    exec_cmd("pngquant --verbose --nofs --force --ext=.png " + output + "/*.png", timeout=None)
    print "advdef -q -z -4 -i 5  " + output + "/*.png"
    exec_cmd("advdef -q -z -4 -i 5  " + output + "/*.png", timeout=None)
    print "gifsicle --batch -O3 -i " + output + "/*.gif"
    exec_cmd("gifsicle --batch -O3 -i " + output + "/*.gif", timeout=None)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='sotoki 0.1')
    if arguments['run']:
        if not bin_is_present("zimwriterfs"):
            sys.exit("zimwriterfs is not available, please install it.")
        # load dump into database
        uuid = str(uuid.uuid1())
        url = arguments['<url>']
        publisher = arguments['<publisher>']
        dump = arguments['--directory']
        database = 'work'
        # render templates into `output`
        templates = 'templates'
        output = os.path.join('work', 'output')
        os.makedirs(output)
        output_tmp= os.path.join('templates', 'tmp')
        os.makedirs(output_tmp)
        os.makedirs(os.path.join(output, 'static', 'images'))
        cores = cpu_count() / 2 or 1
        title, description = grab_title_description_favicon(url, output)
        load_user(dump, templates, database, output, title, publisher, uuid)
        comments(templates, output_tmp, dump, cores, uuid)
        post_type2(templates, output_tmp, dump, cores,uuid)
        posts_links(templates, output_tmp, dump, uuid)
        render_questions(templates, database, output, title, publisher, dump, cores, uuid)
        render_tags(templates, database, output, title, publisher, dump)
        del_redis_keys(uuid)
        #remove tmp files
        shutil.rmtree(output_tmp)
        # copy static
        optimize(os.path.join(output, 'static', 'images'))
        copy_tree('static', os.path.join('work', 'output', 'static'))
        create_zims(title, publisher, description)

