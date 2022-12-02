import itertools
import subprocess

class SimBox(object):
    """The SimBox contains all information about a simulation box, that can be packed into a PDB-File later turned into the topology of the simulation.

    Args:
        object ([type]): [description]
    """

    def __init__(self, Project, total_number_molecules, box_side_length, chi_water, water_name, mixname):
        """The simulation box is initialized just taking parameters regarding the box into account, not simulation process related ones.

        Args:
            Project (object): The Project-Object holds all information about the Project.
            total_number_molecules (int): The number of molecules that should be inside the box.
            box_side_length (int): The length of the side of the simulation box in angstroms.
            chi_water (float): The water mole fraction of the simulation box.
            water_name (str): The name of the water PDB-File, auto-assigned when setting water=True as instancing a molecule.
            mixname (str): A auto-generated string containing certain box-characteristics, that can be used to access the boxes later.
        """

        self.boxdir = Project.boxdir
        self.names = Project.names
        self.atnums = Project.atnums
        self.molras = Project.molras
        self.totnum = total_number_molecules
        self.box_side_length = box_side_length
        self.chi_water = chi_water
        self.mixname = mixname
        self.moldir = Project.moldir
        self.waname = water_name
        self.path = f'{self.boxdir}/mixture_{mixname}.pdb'


    def pack(self):
        """Packs the simulation box using packmol by auto-generating a packmol input file.
        """

        mixname = self.mixname
        folder = self.boxdir

        if self.chi_water is not None:
            totnum = int(self.totnum)
            chi_water = float(self.chi_water)
            chi_water_num_t = chi_water * totnum
            chi_water_num = int(chi_water_num_t)
            re_num = totnum-chi_water_num
            mol_ra_li = list(map(int, self.molras))
            boxsize = int(self.box_side_length)
            store = self.moldir
            numdict = dict(zip(self.names, mol_ra_li))
            a = sum(mol_ra_li)

            with open(f'{folder}/mixture_{mixname}.inp', 'w') as prepmix:
                prepmix.write(f'tolerance 2.0\nfiletype pdb\noutput {folder}/mixture_{mixname}.pdb\n\n')
            with open(f'{folder}/mixture_{mixname}.inp', 'a') as prepmix:
                for x in numdict:
                    m = int(round(numdict[x])*(re_num/a))
                    prepmix.write(f'structure {store}/{x}.pdb\n\tnumber {m}\n\tinside box 0. 0. 0. {boxsize}. {boxsize}. {boxsize}.\nend structure\n\n')
                if chi_water_num == 0: 
                    molnum_t = [numdict[x]*(re_num/a) for x in numdict]
                    molnum = [int(i) for i in molnum_t]
                    stack = [0]
                    b=0
                    for x,y in zip (molnum, self.atnums):
                        stack.append(sum(stack[b:])+x*y)
                        b+=1
                    del stack[-1]
                    self.stack=stack
                    self.molnum=molnum
                else:
                    prepmix.write(f'structure {store}/{self.waname}.pdb\n\tnumber {chi_water_num}\n\tinside box 0. 0. 0. {boxsize}. {boxsize}. {boxsize}.\nend structure\n\n')
                    molnum_t = [numdict[x]*(re_num/a) for x in numdict]+[chi_water_num]
                    molnum = [int(i) for i in molnum_t]
                    stack = [0]
                    b=0
                    for x,y in zip (molnum, self.atnums):
                        stack.append(sum(stack[b:])+x*y)
                        b+=1 
                    names_new = self.names+[f'{self.waname}']
                    atnum_new = self.atnums+[4]
                    self.names=names_new
                    self.atnum=atnum_new
                    self.stack=stack
                    self.molnum=molnum
            with open(f'{folder}/mixture_{mixname}.inp', 'a') as prepmix:
                prepmix.write(f'add_box_sides 1.0')
            arguments = f'packmol < {folder}/mixture_{mixname}.inp'
            _ = subprocess.run(arguments, shell= True)

        else:
            totnum = int(self.totnum)
            mol_ra_li = list(map(int, self.molras))
            boxsize = int(self.box_side_length)
            store = self.moldir
            numdict = dict(zip(self.names, mol_ra_li))
            a = sum(mol_ra_li)

            with open(f'{folder}/mixture_{mixname}.inp', 'w') as prepmix:
                prepmix.write(f'tolerance 2.0\nfiletype pdb\noutput {folder}/mixture_{mixname}.pdb\n\n')
            with open(f'{folder}/mixture_{mixname}.inp', 'a') as prepmix:
                for x in numdict:
                    m = int(round(numdict[x])*(totnum/a))
                    prepmix.write(f'structure {store}/{x}.pdb\n\tnumber {m}\n\tinside box 0. 0. 0. {boxsize}. {boxsize}. {boxsize}.\nend structure\n\n')
                molnum_t = [numdict[x]*(totnum/a) for x in numdict]
                molnum = [int(i) for i in molnum_t]
                stack = [0]
                b=0
                for x,y in zip (molnum, self.atnums):
                    stack.append(sum(stack[b:])+x*y)
                    b+=1
                del stack[-1]
                self.stack=stack
                self.molnum=molnum
            with open(f'{folder}/mixture_{mixname}.inp', 'a') as prepmix:
                prepmix.write(f'add_box_sides 1.0')
            arguments = f'packmol < {folder}/mixture_{mixname}.inp'
            _ = subprocess.run(arguments, shell= True)


    def conectGenerator(self):
        """Writes the CONECT entry for the PDB-File that has been packed with packmol.
        """
        
        store = self.moldir
        names = self.names
        molnum = self.molnum
        numdict = dict(zip(names, molnum))
        at = self.atnums
        atnum = dict(zip(names, at))
        stack = self.stack
        stackdict = dict(zip(names, stack))
        path = self.path
        numcomp = (len(names)-1)
    
        def grabCONECTfromPDB(solvent):
            """Gets all CONECT entry lines from the PDB-Files of the molecules as a list of integers.

            Args:
                solvent (str): Name of the molecule, that is used to get its PDB-File.

            Returns:
                list: A list with the CONECT values of the PDB-File.
            """
            solvent_li = []
            for line in open(f'{store}/{solvent}.pdb'):
                if line[0] == 'C':
                    data = line
                    solvent_li.append(data)
            solvent_str = ''.join(solvent_li)
            li_grabCONECTfromPDB = [int(s) for s in solvent_str.split() if s.isdigit()]
            return li_grabCONECTfromPDB
        # Create a dict with molecule names as key and integers from the string as a list as values
        CONECTfromPDB = dict.fromkeys(names)
        for key, _ in CONECTfromPDB.items():
            CONECTfromPDB[key] = grabCONECTfromPDB(key)
        def addtoCONECTList(CONECTdict,solvent,stackdict):
            """Creates all CONECT values for the simulation box by repeating the CONECT values times the number of the respective molecule and increasing the CONECT values.

            Args:
                CONECTdict (dict): Contains the CONECT values for every molecule.
                solvent (str): Name of the molecule.
                stackdict (dict): Contains the number of atoms that are in the box before a molecule is added as each entry timewise.

            Returns:
                list: A list with all CONECT values in increasing order.
            """
            li_AddtoCONECTList = []
            m = solvent
            n = numdict[m]
            for _ in range(n):
                li_AddtoCONECTList.append([])
            for x in range(n):
                li_AddtoCONECTList[x].append([(f+((atnum[m])*x)+stackdict[m]) for f in CONECTdict[m]])
            return li_AddtoCONECTList
        # Create a dict with molecule names as key and CONECT lists as value
        FinalCONECT = dict.fromkeys(names)
        for key, _ in CONECTfromPDB.items():
            for _ in range(numcomp):
                FinalCONECT[key] = addtoCONECTList(CONECTfromPDB,key,stackdict)
        def getFrame(solvent):
            """Gets the frame of the CONECT values in the PDB-Files of the molecules.

            Args:
                solvent (str): Name of the molecule.

            Returns:
                list: Frame as list of integers that describe the last column that contains a value in the CONECT section of the PDB.
            """
            solvent_li_getFrame = []
            for line in open(f'{store}/{solvent}.pdb'):
                f2 = line[12:16].strip()
                f3 = line[17:21].strip()
                f4 = line[22:26].strip()
                f5 = line[27:31].strip()
                if line[0] == 'C' and f5.isdigit() == True:
                    data = 5  
                    solvent_li_getFrame.append(data)
                elif line[0] == 'C' and f4.isdigit() == True:
                    data = 4  
                    solvent_li_getFrame.append(data)
                elif line[0] == 'C' and f3.isdigit() == True:
                    data = 3    
                    solvent_li_getFrame.append(data)
                elif line[0] == 'C' and f2.isdigit() == True:
                    data = 2     
                    solvent_li_getFrame.append(data)
                else:
                    data = 1            
            return solvent_li_getFrame
        CONECTFrame = dict.fromkeys(names)
        for key, _ in CONECTFrame.items():
            CONECTFrame[key] = getFrame(key)
        def flattenLists(solvent):
            """Flattens a list twice.

            Args:
                solvent (str): Name of the molecule.

            Returns:
                list: Flattened list.
            """
            m = solvent
            li1=list(itertools.chain.from_iterable(FinalCONECT[m]))
            li2=list(itertools.chain.from_iterable(li1))
            return li2
        FinalCONECTflat = dict.fromkeys(names)
        for key, _ in FinalCONECTflat.items():
            FinalCONECTflat[key] = flattenLists(key)
        def createFrameList(solvent,CONECTFrame):
            """Creates a list with all frame information for the simulation box.

            Args:
                solvent (str): Name of the molecule.
                CONECTFrame (dict): Contains the CONECT frame for each molecule.

            Returns:
                list: List with all CONECT frame values.
            """
            m = solvent
            solvent_li_createFrameList = CONECTFrame[m]
            solvent_li1_createFrameList = []
            for _ in itertools.repeat(None, numdict[m]):
                solvent_li1_createFrameList.extend(solvent_li_createFrameList)
            return solvent_li1_createFrameList
        CONECTFrameCombined = dict.fromkeys(names)
        for key, _ in CONECTFrameCombined.items():
            CONECTFrameCombined[key] = createFrameList(key,CONECTFrame)
        def cleanNullTerms(DICT):
            """Removes all None values out of a dictionary.

            Args:
                DICT (dict): Dictionary to that the function will be applied.

            Returns:
                dict: Dictionary with removed None entries.
            """
            return {
                    k:v
                  for k, v in DICT.items()
                  if v is not None
            }
        CONECTFrameCombinedC = cleanNullTerms(CONECTFrameCombined)
        FinalCONECTflatC = cleanNullTerms(FinalCONECTflat)
        def mergeListsFromDict(DICT):
            """Merges lists from dictionaries into one list.

            Args:
                DICT (dict): Dictionary to that the function will be applied.

            Returns:
                list: List from lists in the dictionary.
            """
            li = []
            for key, _ in DICT.items():
                li.extend(DICT[key])
            return li
        FrameList = mergeListsFromDict(CONECTFrameCombinedC)
        CONECTList = mergeListsFromDict(FinalCONECTflatC)
        def writeCONECTtoTXT(FrameList,CONECTList):
            """Writes the CONECT section of the simulation box PDB-File line by line.

            Args:
                FrameList (list): List with the frame the simulation box PDB-File should have.
                CONECTList (list): List with the CONECT entries  the simulation box PDB-File should have.
            """
            n = 0
            fin = open(f'{path}', 'rt')
            data = fin.read()
            data = data.replace('END\n', '')
            fin.close()
            fin = open(f'{path}', 'wt')
            fin.write(data)
            fin.close()
            with open (f'{path}', 'a') as mix:
                for f, _ in itertools.zip_longest (FrameList, CONECTList):
                    if f == 1:
                        mix.write(f'CONECT{CONECTList[n]:5}')
                        n+=1
                    elif f == 2:
                        mix.write(f'CONECT{CONECTList[n]:5}{CONECTList[n+1]:5}\n')
                        n+=2
                    elif f == 3:
                        mix.write(f'CONECT{CONECTList[n]:5}{CONECTList[n+1]:5}{CONECTList[n+2]:5}\n')
                        n+=3
                    elif f == 4:
                        mix.write(f'CONECT{CONECTList[n]:5}{CONECTList[n+1]:5}{CONECTList[n+2]:5}{CONECTList[n+3]:5}\n')
                        n+=4
                    elif f == 5:
                        mix.write(f'CONECT{CONECTList[n]:5}{CONECTList[n+1]:5}{CONECTList[n+2]:5}{CONECTList[n+3]:5}{CONECTList[n+4]:5}\n')
                        n+=5
                    else:
                        pass
                mix.write('END')
            return None
        writeCONECTtoTXT(FrameList, CONECTList)