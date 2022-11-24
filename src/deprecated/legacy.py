# -*- coding: utf-8 -*-
"""
Legacy
======

Created on Thu Nov 12 14:59:42 2020

@author: hcarv
"""

    def combinatorial_deprecated(self, shells, labels=None):
        """Class to generate combinatorial encoding of shells into states. Discretization is made based on combination of shells. 
        Not sensitive to which shell a given molecule is at each frame. Produces equal-sized strings for all parameters.
        Values of shell boundaries are given by shells array.
        Static methods defined in class *Functions* class are employed here."""

        msm_combinatorial_f={}
        
        for parameter, data_dict in self.features.items():
            
            disc_path='{}/discretized-combinatorial-{}-{}.npy'.format(self.results, self.name, parameter)
            raw_data=[] #data_dict may contain one or more .npy objects
            for i in data_dict:
                
                i_arr=np.load(i)
                raw_data.append(i_arr)
            rep, frames, ref, sel=np.shape(raw_data)
            raw_reshape=np.asarray(raw_data).reshape(rep*frames, ref*sel)
            
            if ref == 1:
                RAW=pd.DataFrame(raw_reshape)
                RAW.columns=RAW.columns + 1
            else:
                RAW=pd.DataFrame()
                RAW_split=np.split(raw_reshape, ref, axis=1)
                for ref in RAW_split:
                    RAW_ref=pd.DataFrame(ref)
                    RAW_ref.columns=RAW_ref.columns + 1
                    RAW=pd.concat([RAW, RAW_ref], axis=0)
            
            state_df=tools.Functions.state_mapper(shells, array=RAW.values, labels=labels) #Call to Functions class
            np.save(disc_path, state_df.values)
            msm_combinatorial_f[parameter]={'discretized':[disc_path]}
        
        return msm_combinatorial_f