Code for ICML 2019 paper on [Fast and Simple Natural-Gradient Variational Inference with Mixture of Exponential-family Approximations](https://arxiv.org/abs/1906.02914)

* To-do List:
  * added a [poster](https://github.com/yorkerlin/VB-MixEF/blob/master/poster_workshop.pdf) and a technical [report](https://arxiv.org/abs/1910.13398) about the gradient identities used in the paper (To appear at the ICML workshop on Stein's method) [done]
  * added a [poster](https://github.com/yorkerlin/VB-MixEF/blob/master/poster_main.pdf) of the main paper.  [done]  
  * [skewness] added a Matlab [implementation](https://github.com/yorkerlin/VB-MixEF/tree/master/src/matlab/skewness) for the toy example using skew Gaussian and exponentially modified Gaussian. [done] 
  * [multi-modality] added a Matlab [implementation](https://github.com/yorkerlin/VB-MixEF/tree/master/src/matlab/multimodality) for the toy example using MoG. The implementation is based on this [repo](https://github.com/TimSalimans/LinRegVB) [done]  For an efficient MOG implementaiton, please see this [repo](https://github.com/yorkerlin/iBayesLRule)
  * [heavy tails] To add a Matlab implementation for BLR using t-distribution and  symmetric normal inverse Gaussian
  * To add a Python implementation for Vadam extensions
