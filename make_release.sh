#!/bin/bash
set -u
if [[ $# -ne 1 ]]
then
	echo "You shoud profide version number as argument"
  exit 1
fi
sed -i "s/version='[^']*',/version='$1',/" setup.py
#update version in sotoki/sotoki.py
git add setup.py
git commit -m "Make release of $1 version"
git tag -a $1 -m "Make release of $1 version"

#we build
python setup.py sdist bdist_wheel
twine upload dist/*

git push -u origin master
git push origin $1
