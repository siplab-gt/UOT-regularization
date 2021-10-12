# UOT-regularization
Code for "Unbalanced Optimal Transport Regularization for Imaging Problems" by Lee, Bertrand, and Rozell. IEEE Transactions on Computational Imaging.

Abstract:
The modeling of phenomenological structure is a crucial aspect in inverse imaging problems. One emerging modeling tool in computer vision is the optimal transport framework. Its ability to model geometric displacements across an image's support gives it attractive qualities similar to optical flow methods that are effective at capturing visual motion, but are restricted to operate in significantly smaller state-spaces. Despite this advantage, two major drawbacks make it unsuitable for general deployment: (i) it suffers from exorbitant computational costs due to a quadratic optimization-variable complexity, and (ii) it has a mass-balancing assumption that limits applications with natural images. We tackle these issues simultaneously by proposing a novel formulation for an unbalanced optimal transport regularizer that has linear optimization-variable complexity. In addition, we present a general proximal method for this regularizer, and demonstrate superior empirical performance on novel dynamical tracking applications in synthetic and real video.

Formal citation:
Lee, John, Nicholas P. Bertrand, and Christopher J. Rozell. "Unbalanced Optimal Transport Regularization for Imaging Problems." IEEE Transactions on Computational Imaging 6 (2020): 1219-1232.

The three major algorithmic solver contributions of the paper are found at the respective codes:
1) solver_LS_UOT_Beckmann_ADMM.m   -- Figures 1, 2, 3 
2) solver_BPDN_UOT_Beckmann_ADMM.m -- Figures 4, 5
3) solver_RPCA_UOT_Beckman_ADMM.m  -- Figures 6, 7, 8

All benchmarked algorithms (in this paper) are also provided. To effectively apply them, you require CVX along with the pre-packaged SDPT3 solver.
