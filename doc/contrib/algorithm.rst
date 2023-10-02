.. _implement-new::

=========================
Implement a new algorithm
=========================

Criteria
^^^^^^^^

If you want to implement an algorithm and include it in the library, you need
to be aware of the criteria that exists in order to be accepted. In general,
any new algorithm must have:

- A publication with a reasonable number of citations.
- A reference implementation or published inputs/outputs that we can validate
  our version against.
- An implementation that doesn't require thousands of lines of new code, or
  adding new mandatory dependencies.

Of course, any of these three guidelines could be ignored in special cases. On
the other hand, we should prioritize the algorithms that have:

- Larger number of citations
- Common parts that can be reused by other/existing algorithms
- Better proven performance over other similar/existing algorithms


Algorithm wish list
^^^^^^^^^^^^^^^^^^^

Some desired algorithms that are not implemented yet in package can be found
`here <https://github.com/scikit-learn-contrib/metric-learn/issues/13>`_ and
`here <https://github.com/scikit-learn-contrib/metric-learn/issues/205>`_.

How to
^^^^^^

1. First, you need to be familiar with the metric-learn API, so check out the
   :ref:`api-structure` first.
2. Propose in `Github Issues
   <https://github.com/scikit-learn-contrib/metric-learn/issues>`_ the algorithm
   you want to incorporate to get feedback from the core developers.
3. If you get a green light, follow the guidelines on :ref:`contrib-code`
