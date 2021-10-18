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

Triage team
^^^^^^^^^^^

The triage team is composed of community members who have permission
on github to label and close issues. Their work is crucial to improve
the communication in the project and limit the crowding of the issue
tracker.

Similarly to what has been decided in the python project, any contributor
may become a member of the scikit-learn triage team, after showing some
continuity in participating to scikit-learn development (with pull
requests and reviews). Any core developer or member of the triage team
is welcome to propose a scikit-learn contributor to join the triage team.
Other core developers are then consulted: while it is expected that most
acceptances will be unanimous, a two-thirds majority is enough. Every new
triager will be announced in the mailing list. Triagers are welcome to
participate in monthly core developer meetings.

Core developers
^^^^^^^^^^^^^^^

Core developers are community members who have shown that they are
dedicated to the continued development of the project through ongoing
engagement with the community. They have shown they can be trusted to
maintain scikit-learn with care. Being a core developer allows
contributors to more easily carry on with their project related
activities by giving them direct access to the project’s repository and
is represented as being an organization member on the scikit-learn GitHub
organization. Core developers are expected to review code contributions,
can merge approved pull requests, can cast votes for and against merging
a pull-request, and can be involved in deciding major changes to the API.

New core developers can be nominated by any existing core developers.
Once they have been nominated, there will be a vote by the current core
developers. Voting on new core developers is one of the few activities
that takes place on the project’s private management list. While it is
expected that most votes will be unanimous, a two-thirds majority of
the cast votes is enough. The vote needs to be open for at least 1 week.

Core developers that have not contributed to the project (commits or
GitHub comments) in the past 12 months will be asked if they want to
become emeritus core developers and recant their commit and voting
rights until they become active again. The list of core developers,
active and emeritus (with dates at which they became active) is public
on the scikit-learn website.

Decision Making Process
=======================

Decisions about the future of the project are made through discussion
with all members of the community. All non-sensitive project management
discussion takes place on the project contributors’ mailing list and
the issue tracker. Occasionally, sensitive discussion occurs on a
private list.

Scikit-learn uses a “consensus seeking” process for making decisions.
The group tries to find a resolution that has no open objections among
core developers. At any point during the discussion, any core-developer
can call for a vote, which will conclude one month from the call for
the vote. Any vote must be backed by a SLEP. If no option can gather
two thirds of the votes cast, the decision is escalated to the TC,
which in turn will use consensus seeking with the fallback option of
a simple majority vote if no consensus can be found within a month.
This is what we hereafter may refer to as “the decision making process”.

Decisions (in addition to adding core developers and TC membership as
above) are made according to the following rules:

- Minor Documentation changes, such as typo fixes, or addition/correction
  of a sentence, but no change of the scikit-learn.org landing page or
  the “about” page: Requires +1 by a core developer, no -1 by a core
  developer (lazy consensus), happens on the issue or pull request page.
  Core developers are expected to give “reasonable time” to others to give
  their opinion on the pull request if they’re not confident others
  would agree.

- Code changes and major documentation changes require +1 by two core
  developers, no -1 by a core developer (lazy consensus), happens on
  the issue of pull-request page.

- Changes to the API principles and changes to dependencies or supported
  versions happen via a Enhancement proposals (SLEPs) and follows the
  decision-making process outlined above.

- Changes to the governance model use the same decision process outlined
  above.

If a veto -1 vote is cast on a lazy consensus, the proposer can appeal
to the community and core developers and the change can be approved or
rejected using the decision making procedure outlined above.

.. _mlep:

Enchancement proposals (MLEPs)
==============================

For all votes, a proposal must have been made public and discussed
before the vote. Such proposal must be a consolidated document, in
the form of a ‘Metric-Learn Enhancement Proposal’ (MLEP), rather than
a long discussion on an issue. A MLEP must be submitted as a
`Github Discussion
<https://github.com/scikit-learn-contrib/metric-learn/discussions>`_
using the MLEP template. See :ref:`mlep-template`.