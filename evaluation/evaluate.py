import numpy as np
import cv2
from skimage import measure

def evaluarSR(Sb, Rb):
    #Se evalua un volumen Sb respecto a un de referencia Rb.
    #Se dan datos en n?mero de voxeles y en n?mero de blobs (valores absolutos y en porcentaje)
        
    #se halla la intersección: 1=S=FP, 2=R=FN, 3=SyR=TP
    S2R = Sb+2*Rb

    #Resultados por pixels
    #Confusion table per voxels (whole image)
    CTvoxels = confusionTable(S2R)     #TP, FP, FN, TN
    #calculo dice por voxeles
    CTvoxels = np.append(CTvoxels, calcDICE(CTvoxels))

    #Analisis por blobs
    #Labels para agrupaciones
    S2R_bool = S2R.astype(bool) #Convert to logical 1 for all non-zeros entries, False for 0
    S2R_array = S2R_bool*1 #Convert logical to array to be able to convert to UMat
    labelsAgrup = np.zeros(S2R.shape)
    if S2R_array.ndim<3:
        labelsAgrup, numAgrup  = measure.label(S2R_array, connectivity = 2, return_num = True)
    else:
         labelsAgrup, numAgrup  = measure.label(S2R_array, connectivity = 3, return_num = True)
    
    CTAgrup = np.zeros([numAgrup, 5])
    outEval = {}
    NdistribSolape10 = np.zeros(10)
    
    if numAgrup:
        
        # Calculo matriz de confusión por agrupaciones
        for itA in range (0,numAgrup):
            ind = np.argwhere(labelsAgrup == itA + 1)
            S2R_array_label = []
            for index in ind:
                S2R_array_label.append(S2R[tuple(index)])
            CTAgrup[itA,0:4] = confusionTable(np.asanyarray(S2R_array_label))

        #Calculo dice por agrupaciones
        CTAgrup [:,4] = calcDICE(CTAgrup)
        
        #indices de los distinto tipos de agrupación
        ind_extra = np.argwhere((CTAgrup[:,0] == 0) & (CTAgrup[:,2] == 0)) #indice en CTAgrup de agrups extra
        ind_miss  = np.argwhere((CTAgrup[:,0] == 0) & (CTAgrup[:,1] == 0)) #ind de agrups miss
        #Hecho más abajo: ind_SolapeParcial = find(CTAgrup(:,1)) #ind de agrups con solape parcial (todos agrupados)
    
        #Resultados por blobs
        #Absolutos:
        Nmiss   = len(ind_miss);  #agrupaciones tipo miss
        Nextra  = len(ind_extra);   #agrupaciones tipo extra
        #Nsolape = hist(CTAgrup(CTAgrup(:,5)>0, 5),[0.015:.01:1]);   #Hecho con mas detalle en la siguiente l?nea. Calculo del solape parcial a partir del DICE (sin incluir solape 0, que ya se tiene)
        labesDistribSolape, NdistribSolape = indHist(CTAgrup[:,4], np.linspace(0,1,101)) # se obtiene tambien los indices en cada bin
        #Nsolape = [sum(Nmiss+Nextra), Nsolape]; #se le añade el solape 0. Histograma de agrupaciones con solape desde cero (ningun solape) a 1 (total).
        
        #figure(1); hold off; plot([0.5:1:100],NdistribSolape, 'rx-')  #pintar gráfica de solape (distribución)
        
        ##SALIDAS
        #pix: resultados por pixels (del global a comparar)

        outEval['xPix'] = {'CT': CTvoxels[0:4]}
        outEval['xPix'].update({'DICE': CTvoxels[4]})
        #pix: resultados de pixeles (en cada agrupacion)
        outEval['xAgrup'] = {'CT': CTAgrup[:,0:4]}
        outEval['xAgrup'].update({'DICE': CTAgrup[:,4]})
    
    
        #resultados promediados de todas las agrupaciones
        mediaAgrups = np.mean(CTAgrup, axis = 0)
        
        if CTAgrup.shape[0] == 1:
            stdAgrups = np.zeros(5)
        else:
            stdAgrups = np.std(CTAgrup, axis = 0)
    
        if mediaAgrups.shape[0]>1:
            outEval['xAgrup'].update({'CTmed': mediaAgrups[0:4]})
            outEval['xAgrup'].update({'CTstd': stdAgrups[0:4]})
            outEval['xAgrup'].update({'DICEmed': mediaAgrups[4]})
            outEval['xAgrup'].update({'DICEstd': stdAgrups[4]})
        # N: resultados por agrupaciones (del global a comparar)
        outEval['NAgrup'] = {'Nmiss': Nmiss}
        outEval['NAgrup'].update({'Nextra': Nextra})
        outEval['NAgrup'].update({'labelsAgrup': labelsAgrup})
    
    
        for itB in range(0,10):
            NdistribSolape10[itB] = np.sum(NdistribSolape[itB*10 :(itB+1)*10])  #acumulado cada 10
        
        outEval['NAgrup'].update({'NdistribSolapeDICE10': NdistribSolape10})
        outEval['NAgrup'].update({'NdistribSolapeDICE100': NdistribSolape})
    
        outEval['NAgrup'].update({'Ntotal': np.sum(NdistribSolape)})
    
        #indices de los distintos tipos de agrupaciones. Para un análisis más
        #fino, habrá que crear otra función más específica
        outEval['indAgrup'] = {'extra': ind_extra}
        outEval['indAgrup'].update({'miss': ind_miss})
        outEval['indAgrup'].update({'labesDistribSolape': labesDistribSolape})
        
    else:
        outEval = []
        
    return outEval

def confusionTable(Sy2R):
    ind = {}
    ind['T_FP'] = np.argwhere(Sy2R == 1)
    ind['T_FN'] = np.argwhere(Sy2R == 2)
    ind['T_TP'] = np.argwhere(Sy2R == 3)
    
    T = np.zeros(4)
    T[0] = ind['T_TP'].shape[0]                       # TP
    T[1] = ind['T_FP'].shape[0]                       # FP
    T[2] = ind['T_FN'].shape[0]                       # FN
    if Sy2R.ndim <2:
        T[3] = Sy2R.shape[0]*1 - np.sum(T[0:3]) # TN
    elif Sy2R.ndim<3:
        T[3] = Sy2R.shape[0]*Sy2R.shape[1] - np.sum(T[0:3]) # TN
    else:
        T[3] = Sy2R.shape[1]*Sy2R.shape[2] - np.sum(T[0:3]) # TN
    
    return T

def calcDICE(CT):
    if CT.ndim <2:
        dice =  2*CT[0]/(2*CT[0]+CT[1]+CT[2])
    else:
        dice =  2*CT[:,0]/(2*CT[:,0]+CT[:,1]+CT[:,2])
    return dice

def indHist(vector, bins):
    
    distribIndices = {}
    histogram = np.zeros(len(bins)-1)
    for itR in range(0,len(bins)-2):
        distribIndices[itR] = np.argwhere((vector>=bins[itR]) & (vector<bins[itR+1]))
        histogram[itR] = len(distribIndices[itR])
    
    distribIndices[len(bins)-2] = np.argwhere((vector>=bins[itR+1]) & (vector<=bins[itR+2]))
    histogram[len(bins)-2] = len(distribIndices[len(bins)-2])
    
    return distribIndices, histogram