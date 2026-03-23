.. _governance:

===========================================
Metric learn governance and decision-making
===========================================

The purpose of this document is to formalize the governance process used by
the metric-learn project, to clarify how decisions are made and how the
various elements of our community interact. This document establishes a
decision-making structure that takes into account feedback from all
members of the community and strives to find consensus, while avoiding
any deadlocks.

This is a meritocratic, consensus-based community project. Anyone with
an interest in the project can join the community, contribute to the
project design and participate in the decision making process. This
document describes how that participation takes place and how to set
about earning merit within the project community.

Roles and Responsibilities
==========================

Contributors
^^^^^^^^^^^^

Contributors are community members who contribute in concrete ways to
the project. Anyone can become a contributor, and contributions can
take many forms – not only code – as detailed in the contributors guide.

Core developers
^^^^^^^^^^^^^^^

Core developers are community members who have shown that they are
dedicated to the continued development of the project through ongoing
engagement with the community. They have shown they can be trusted to
maintain metric-learn with care. Being a core developer allows
contributors to more easily carry on with their project related
activities by giving them direct access to the project’s repository and
is represented as being an organization member on the metric-learn GitHub
organization. Core developers are expected to review code contributions,
can merge approved pull requests, can cast votes for and against merging
a pull-request, can label and close issues, and can be involved in
deciding major changes to the API.


Decision Making Process
=======================

Decisions about the future of the project are made through discussion
with all members of the community. Metric-learn uses a “consensus seeking”
process for making decisions.

The group tries to find a resolution that has no open objections among
core developers. At any point during the discussion, any core-developer
can call for a vote.

Decisions are made according to the following rules:

- Minor Documentation changes, such as typo fixes, or addition/correction
  of a sentence, requires +1 by a core developer, no -1 by a core
  developer (lazy consensus), happens on the issue or pull request page.
  Core developers are expected to give “reasonable time” to others to give
  their opinion on the pull request if they’re not confident others
  would agree.

- Code changes and major documentation changes require +1 by two core
  developers, no -1 by a core developer (lazy consensus), happens on
  the issue of pull-request page.

- Changes to the API principles and changes to dependencies or supported
  versions happen via a Enhancement proposals (MLEPs) and follows the
  decision-making process outlined above.

If a veto -1 vote is cast on a lazy consensus, the proposer can appeal
to the community and core developers and the change can be approved or
rejected using the decision making procedure outlined above.

.. _mlep:

Enchancement proposals (MLEPs)
==============================

For vote on API changes, a proposal must have been made public and discussed
before the vote. Such proposal must be a consolidated document, in
the form of a ‘Metric-Learn Enhancement Proposal’ (MLEP), rather than
a long discussion on an issue. A MLEP must be submitted as a
`Github Discussion
<https://github.com/scikit-learn-contrib/metric-learn/discussions>`_
using the MLEP template. See :ref:`mlep-template`.