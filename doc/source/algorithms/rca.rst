Relative Components Analysis (RCA)
=====================================
RCA is developed for unsupervised learning using equivalence relations. It uses only closed form expressions of the data, based on an information theoretic basis. RCA is a simple and efficient algorithm for learning a full ranked Mahalanobis distance metric, which is constructed based on a weighted sum of in-class covariance matrices. It applies a global linear transformation to assign large weights to relevant dimensions but low weights to irrelevant dimensions. In RCA, those relevant dimensions are estimated using the term chunklets, which are subsets of points that are known to belong to the same although unknown class. A distance metric that represents the somewhat natural variance of the original unknown class information can then be trained using RCA. 

Example Code
------------------

References
------------------
`Adjustment learning and relevant component analysis <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.2871&rep=rep1&type=pdf>`_ Noam Shental, et al.