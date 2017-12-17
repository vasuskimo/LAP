# LAP
# Unsupervised Classification using Localized Archetypal Pivots (LAP)
## What is LAP? 
LAP is a density-based unsupervised classification algorithm which is very different from clustering or archetypal analysis.
LAP not only tries to find the localities of higher density, but also attempts to find a localized archtypal pivot, that is representative of the locality.

## Why LAP? 
To understand the context, let us review some popular unsupervised algorithms: K-Means, DBSCAN and archetypal analysis. 
K-Means works by specifying the number of clusters and then attempts to find centroids that define a cluster. 
The biggest flaw of this type of clustering is that you need to specify the number of clusters and based on the starting point, the clusters formed would be different each time. 
DBSCAN works by finding dense clusters and separating those that are far away from this dense cluster. 
DBSCAN also falls short because it tries to create uniformly sized clusters. 
Archetypal analysis attempts to find archetypal examples that explain the entire data. 
These extreme examples are sensitive to outliers and are non-local by nature. 
For many learning tools, such as regression, non-locality is a deal breaker. 

Unsupervised Classification using Localized Archetypal Pivots (LAP) overcomes the problems of non-locality and being sensitive to outliers, by trying to find an archetypal pivot that is closest to the most number of points in a locality, rather than the entire global space. 
The archetypal pivot differs from the archetype of the Archetypal analysis, in that the archetype in the latter case form a convex combination of the members, but the archetypal pivot could be either an affine or convex combination of the members of the class.

## How does LAP Work? 
Loop until there are no more points in your space or until there are no more pivots:
* Select a random set of k points from your space of m data points.
* Find the pivot point p, that is close (i.e., distance less than a threshold t), to most points in the set. 
  * The pivot point could be either in the sample set or in the rest of the global set, which is why it is in convex combination or affine combination w.r.t. the sample set.
* Find all points from the global set, that are close to the pivot point.
* Find the target function that explains all these points.
* Separate the unlabeled class from the rest and give a unique automatic label to this class.
* Finally, perform one of the following:
  * Classify the rest of the points to best fit, into one of the classes using a SoftMax function. (or) 
  * Label the rest of the points manually into one of the classes identified. (or) 
  * Fit some of the points into one of the classes and classify the rest as outliers.
