.. _release:
 
========================
Publishing a new release
========================

This section has the information needed to do different tasks related to
releases. This task is usually performed by core developers but it
might be requested to other developers in the future.

Before the release
==================

1. Make sure that all versions numbers are updated in ``metric_learn/_version.py```
   and in ``doc/conf.py```.
2. Make sure that the final year date in doc/conf.py after copytight is the right
   one (e.g 2022)
3. Do the release summary indicating: Changes, Features, Bug Fixes, Maintenance,
   and any relevant information .

Pre-release
===========


1. Create branch v0.1.0-release from master that includes all changes mentioned above, e.g. version number, etc.
2. Turn this branch into protected branch.
3. Create a release v0.1.0-rc0 ("rc" stands for release candidate) from this branch v0.1.0-release via GitHub UI `here <https://github.com/brianray/data-describe/releases>`_ and mark it as pre-release.
4. Give the community some time (e.g. 1-2 weeks) to test the release candidate.
5. If there are any bug fixes, we push to v0.1.0-release branch and then release v0.1.0-rc1.
6. Once we are confident, we create a stable release v0.1.0, build the package wheels, and then publish the package v0.1.0 to PyPI (instructions below)

Release
=======

1. On Github, click on "draft a new release" button here:
   https://github.com/scikit-learn-contrib/metric-learn/releases, draft the release
   (you can look in the link at the commits made since the last release to help write
   the message of the release). Then click on "Publish release" (it will automatically
   add the files needed).
2. Run the following commands in the repo in a local terminal: this will push the repo
   to PyPi (you need an account on PyPi)

   .. code-block:: bash

      python3 setup.py sdist
      python3 setup.py bdist_wheel
      twine upload dist/*

3. Normally after a few minutes you should see that the badge on the ``README.rst`` gets
   updated with the new version, also the version is available if you search for it
   on PyPi.


Publish the docs
================

1. Make sure you can build the docs. Follow the steps from the section
   :ref:`building-the-docs`.
2. Once you've built the docs, copy all the content inside ``doc/_build/html`` into a temporary
   folder.
3. Checkout to ``gh-pages`` branch.
4. Delete everything except the ``.git`` and the ``.nojekyll`` (`reference <https://github.blog/2009-12-29-bypassing-jekyll-on-github-pages/>`_)
5. Paste the content of ``doc/_build/html`` in the root directory of this branch.
6. Push your changes and create a PR.
7. Once the PR is merged, the website will be automatically updated with the latest changes

.. note::

  This process should be automated. Feel free to create a PR  for this
  `open issue <https://github.com/scikit-learn-contrib/metric-learn/issues/250>`_.

Update Pypi and Conda
=====================

1. If requirements, license, etc have not changed. PR is automatically created in the `feedstock repository <https://github.com/conda-forge/metric-learn-feedstock>`_ for conda-forge.
2. Otherwise, one dev can edit it and directly push to the PR (see `here <https://conda-forge.org/docs/maintainer/updating_pkgs.html>`.). Then merging the PR (which requires to be identified as a maintainer of the feedstock).
3. If it is not visible `here <https://anaconda.org/conda-forge/metric-learn>`_ a dev can do an empty commit to master to trigger the update.