============
Contributing
============

This project is a community effort, and everyone is welcome
to contribute.

The project is hosted on https://github.com/scikit-learn-contrib/metric-learn/

The decision making process and governance structure of metric-learn
is laid out in the governance document: :ref:`governance`.

Metric-learn is somewhat selective when it comes to adding new
algorithms, and the best way to contribute and to help the project
is to start working on known issues.

In case you experience issues using this package, do not hesitate to
submit a ticket to the `GitHub issue tracker
<https://github.com/scikit-learn-contrib/metric-learn/issues>`_.
You are also welcome to post feature requests or pull requests.

Our community, our values
=========================

We are a community based on openness and friendly, didactic, discussions.

We aspire to treat everybody equally, and value their contributions. We
are particularly seeking people from underrepresented backgrounds in Open
Source Software and scikit-learn in particular to participate and contribute
their expertise and experience.

Decisions are made based on technical merit and consensus.

Code is not the only way to help the project. Reviewing pull requests,
answering questions to help others on mailing lists or issues, organizing
and teaching tutorials, working on the website, improving the documentation,
are all priceless contributions.

We abide by the principles of openness, respect, and consideration of others
of the Python Software Foundation: https://www.python.org/psf/codeofconduct/

Ways to contribute
==================

There are many ways to contribute to metric-learn, with the most common
ones being contribution of code or documentation to the project. Improving
the documentation is no less important than improving the library itself.
If you find a typo in the documentation, or have made improvements, do not
hesitate to send an email to the mailing list or preferably submit a GitHub
pull request. Full documentation can be found under the doc/ directory.

But there are many other ways to help. In particular helping to improve,
triage, and investigate issues and reviewing other developers’ pull
requests are very valuable contributions that decrease the burden on the
project maintainers.

Another way to contribute is to report issues you’re facing, and give a
“thumbs up” on issues that others reported and that are relevant to you.
It also helps us if you spread the word: reference the project from your
blog and articles, link to it from your website, or simply star to say
“I use it”:

In case a contribution/issue involves changes to the API principles or
changes to dependencies or supported versions, it must be backed by a
:ref:`mlep`, where a MLEP must be submitted as a new
`Github Discussion
<https://github.com/scikit-learn-contrib/metric-learn/discussions>`_
using the :ref:`mlep-template` and follows the decision-making process
outlined in metric-learn
:ref:`governance`.

Submitting a bug report or a feature request
============================================

We use GitHub issues to track all bugs and feature requests; feel free
to open an issue if you have found a bug or wish to see a feature
implemented.

In case you experience issues using this package, do not hesitate to
submit a ticket to the `Bug Tracker
<https://github.com/scikit-learn-contrib/metric-learn/labels/bug>`_.
You are also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the following
rules before submitting:

- Verify that your issue is not being currently addressed by other
  `issues <https://github.com/scikit-learn-contrib/metric-learn/issues>`_
  or `pull requests
  <https://github.com/scikit-learn-contrib/metric-learn/pulls>`_.

- If you are submitting an algorithm or feature request, please
  verify that the algorithm fulfills our new algorithm requirements.

- If you are submitting a bug report, we strongly encourage you to
  follow the guidelines in How to make a good bug report.

How to make a good bug report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you submit an issue to `Github
<https://github.com/scikit-learn-contrib/metric-learn/issues>`_, please
do your best to follow these guidelines! This will make it a lot easier
to provide you with good feedback:

- The ideal bug report contains a **short reproducible code snippet**,
  this way anyone can try to reproduce the bug easily (see `this
  <https://stackoverflow.com/help/minimal-reproducible-example>`_
  for more details). If your snippet is longer than around 50 lines,
  please link to a `gist <https://gist.github.com/>`_ or a github repo.

- If not feasible to include a reproducible snippet, please be specific
  about what **metric learners and/or functions are involved and the
  shape of the data**.

- Please include your **operating system type and version number**, as
  well as your **Python, metric-learn, scikit-learn, numpy, and scipy**
  versions. This information can be found by running the following
  code snippet:

.. code-block::

  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import sklearn; print("Scikit-Learn", sklearn.__version__)
  import metric_learn; print("Metric-Learn", metric_learn.__version__)

- Please ensure all code snippets and error messages are formatted
  in appropriate code blocks. See `Creating and highlighting code
  blocks <https://docs.github.com/en/github/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks>`_
  for more details.

.. _contrib-code::

Contributing code
=================

.. note::

  To avoid duplicating work, it is highly advised that you search
  through the issue tracker and the PR list. If in doubt about duplicated
  work, or if you want to work on a non-trivial feature, it’s recommended
  to first open an issue in the issue tracker to get some feedbacks from
  core developers.

  One easy way to find an issue to work on is by applying the “help wanted”
  label in your search. This lists all the issues that have been unclaimed
  so far. In order to claim an issue for yourself, please comment exactly
  `take` on it for the CI to automatically assign the issue to you.

How to contribute
^^^^^^^^^^^^^^^^^

The preferred way to contribute to scikit-learn is to fork the `main
repository <https://github.com/scikit-learn-contrib/metric-learn>`_
on GitHub, then submit a “pull request” (PR).

In the first few steps, we explain how to locally install scikit-learn,
and how to set up your git repository:

1. `Create an account on GitHub <https://github.com/join>`_ if
   you do not already have one.
2. Fork the `project repository
   <https://github.com/scikit-learn-contrib/metric-learn>`_: click
   on the ‘Fork’ button near the top of the page. This creates a copy
   of the code under your account on the GitHub user account. For more
   details on how to fork a repository see `this guide
   <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.
3. Clone your fork of the metric-learn repo from your GitHub account
   to your local disk:

  .. code-block:: bash

    git clone git@github.com:YourLogin/scikit-learn.git  # add --depth 1 if your connection is slow
    cd scikit-learn

4. Install the development dependencies:

  .. code-block:: bash

    pip install numpy scipy scikit-learn pytest matplotlib skggm sphinx shinx_rtd_theme sphinx-gallery numpydoc

5. Install metric-learn in editable mode:

  .. code-block:: bash

    pip install -e .

.. _upstream:

6. Add the ``upstream`` remote. This saves a reference to the main
   metric-learn repository, which you can use to keep your repository
   synchronized with the latest changes:

  .. code-block:: bash

    git remote add upstream https://github.com/scikit-learn-contrib/metric-learn

7. Synchronize your ``main`` branch with the ``upstream/main`` branch,
   more details on `GitHub Docs
   <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_:

  .. code-block:: bash

    git checkout main
    git fetch upstream
    git merge upstream/main

8. Create a feature branch to hold your development changes:

  .. code-block:: bash

    git checkout -b my_feature

  and start making changes. Always use a feature branch. It's good
  practice to never work on the ``main`` branch!

9. Develop the feature on your feature branch on your computer, using Git to
   do the version control. When you're done editing, add changed files using
   ``git add`` and then ``git commit``:

  .. code-block:: bash

    git add modified_files
    git commit

  to record your changes in Git, then push the changes to your GitHub
  account with:

  .. code-block:: bash

    git push -u origin my_feature

10. Follow `these
    <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
    instructions to create a pull request from your fork.


It is often helpful to keep your local feature branch synchronized with the
latest changes of the main scikit-learn repository:

.. code-block:: bash

  git fetch upstream
  git merge upstream/main

Subsequently, you might need to solve the conflicts. You can refer to the
`Git documentation related to resolving merge conflict using the command
line
<https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_.

.. topic:: Learning git:

  The `Git documentation <https://git-scm.com/documentation>`_ and
  http://try.github.io are excellent resources to get started with git,
  and understanding all of the commands shown here.

Pull request checklist
^^^^^^^^^^^^^^^^^^^^^^

Before a PR can be merged, it needs to be approved by two core developers.
Please prefix the title of your pull request with ``[MRG]`` if the
contribution is complete and should be subjected to a detailed review. An
incomplete contribution -- where you expect to do more work before receiving
a full review -- should be prefixed ``[WIP]`` (to indicate a work in
progress) and changed to ``[MRG]`` when it matures. WIPs may be useful to:
indicate you are working on something to avoid duplicated work, request
broad review of functionality or API, or seek collaborators. WIPs often
benefit from the inclusion of a `task list
<https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments>`_ in
the PR description.

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules before marking a PR as ``[MRG]``. The
**bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarises what your
   contribution does. This title will often become the commit message once
   merged so it should summarise your contribution for posterity. In some
   cases "Fix <ISSUE TITLE>" is enough. "Fix #<ISSUE NUMBER>" is never a
   good title.

2. **Make sure your code passes the tests**. The whole test suite can be run
   with `pytest`, if all tests pass, you are ready to push your changes,
   otherwise the CI will detect some tests don't pass later on, you need
   to avoid this.

   Check the :ref:`testing_guidelines` for more details on testing.

3. **Make sure your code is properly commented and documented**, and **make
   sure the documentation renders properly**. To build the documentation, please
   refer to our :ref:`contribute_documentation` guidelines.

4. **Tests are necessary for enhancements to be
   accepted**. Bug-fixes or new features should be provided with
   `non-regression tests
   <https://en.wikipedia.org/wiki/Non-regression_testing>`_. These tests
   verify the correct behavior of the fix or feature. In this manner, further
   modifications on the code base are granted to be consistent with the
   desired behavior. In the case of bug fixes, at the time of the PR, the
   non-regression tests should fail for the code base in the ``main`` branch
   and pass for the PR code.

5. **Make sure that your PR does not add PEP8 violations**. To check the
   code that you changed, you can run the following command (see
   :ref:`above <upstream>` to set up the ``upstream`` remote):

   .. code-block:: bash

    git diff upstream/main -u -- "*.py" | flake8 --diff

   or `make flake8-diff` which should work on unix-like system.

   You can also run the following code while you develop, to check your that
   the coding style is correct:

   .. code-block:: bash

    flake8 --extend-ignore=E111,E114 --show-source --exclude=venv

.. _testing_guidelines:

Testing guidelines
^^^^^^^^^^^^^^^^^^

Follow these simple guidelines to test your new feature/module:

1. Place all yout tests in the `test/` directory. All new tests
   must be under a new file named `test_my_module_name.py`. Discuss
   in your pull request where these new tests should be put in the
   package later on.
2. All test methods inside this file must start with the `test_`
   prefix, so pytest can detect and execute them.
3. Use a good naming for your tests that matches what it actually
   does.
4. Comment each test you develop, to know in more detail what it
   is intended to do and check.
5. Use pytest decorators. The most important one is `@pytest.mark.parametrize`.
   That way you can test your method with different values without
   hard-coding them.
6. If you need to raise a `Warning`, do a test that verifies that
   the warning is being shown. Same for `Errors`. Some examples might
   be warnings about a default configuration, a wrong input, etc.

.. _building-the-docs:

Building the docs
^^^^^^^^^^^^^^^^^

To build the docs is always recommended to start with a fresh virtual
environment, to make sure that nothing is interfering with the process.

1. Create a new Python virtual environment named `venv`

  .. code-block:: bash

    python3 -m venv venv

2. Install all dependencies needed to render the docs

  .. code-block:: bash

    pip3 install numpy scipy scikit-learn pytest matplotlib skggm sphinx shinx_rtd_theme sphinx-gallery numpydoc

3. Install your local version of metric_learn into the virtual environment,
   from the root directory.

  .. code-block:: bash

    pip3 install -e .

5. Go to your doc directory and complies

  .. code-block:: bash

    cd doc
    make html

6. Open the `index.html` file inside `doc/_build/html`