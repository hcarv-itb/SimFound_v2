# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:27:16 2020

@author: hca
"""

#Variants with normalized statdist
# =============================================================================
# pi_norm=pd.DataFrame()
# for name, model in msm_dict.items():
#     print(name)
#     if model['bayesMSM'] != None:
#         model_prior=model['bayesMSM'] 
#         scheme, feature, parameter=name.split('-')
#         
#         if scheme != 'combinatorial':        
#             prior_statdist=pi.loc[:, (scheme, feature, parameter)].values
#             states=pi.loc[:, (scheme, feature, parameter)].index.values                        
#             norm_statdist=np.asarray(prior_statdist/(states**2), dtype=np.float64)
#             
#             model['bayesMSM-rNormalized']=model['instance'].bayesMSM(lag, 
#                                     variant='rNormalized', statdist_model=norm_statdist)
#     else:
#         model['bayesMSM-rNormalized']=None
# 
#     #msm_dict[name].append(model_norm)   
#     
# pi_norm=pd.DataFrame()
# 
# for name, model in msm_dict.items():
#     print(name)
#     pi_model=model['instance'].stationaryDistribution(name, model['bayesMSM-rNormalized'])
#     pi_norm=pd.concat([pi_norm, pi_model], axis=1)
# 
# pi.to_csv(f'{results}/stationary_distributions_norm.csv')
# 
# 
# for scheme, v in disc.items():
#     for feature in v.keys(): 
#         if scheme != 'combinatorial':
#             pi_norm.xs((scheme, feature), axis=1).plot(linestyle='-', marker='o') #kind='bar', stacked=True)
#             plt.xlabel('State Index')
#             plt.ylabel('Stationary Distribution')
#             plt.title(f'{scheme}-{feature}-norm')
#             plt.savefig(f'{results}/stationary_distribution-{scheme}-{feature}-norm.png', bbox_inches='tight', dpi=600)
#             plt.show()
# =============================================================================
