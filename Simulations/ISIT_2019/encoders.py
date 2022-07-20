import numpy as np
# import sparse
# import dask.array as da
# from dask import delayed
from scipy.sparse import csr_matrix, lil_matrix

def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 

def get_union(dict_of_lists):
    union_list = []
    for list_key in dict_of_lists:
        union_list = Union(union_list,dict_of_lists[list_key])
    return union_list

def RS(m,c,delta):
    #Returns Robust Soliton CDF for given parameters
    S=c*np.log(m/delta)*np.sqrt(m)
    rho=0.0
    tau=0.0
    r=round(m/S)
    u=np.zeros(m)
    for d_iter in range(m):
        d=float(d_iter+1)
        if d==1:
            rho=1/float(m)
        else:
            rho=1/(d*(d-1))
        if d>=1 and d<r:
            tau=1/(d*r)
        elif d==r:
            tau=np.log(S/delta)/r
        else:
            tau=0
        u[d_iter]=tau+rho
    u=u/np.sum(u)
    return u

def mds_encoder(mat,num_old,num_new):
    rows = mat.shape[0]
    subrows=int(rows/num_old)
    eyemat = np.eye(subrows)
    mdsmat = np.random.normal(size=(num_new,num_old)) #Mapping at client corresponding to (num_new,num_old) MDS code
    genmat_list=[] #Generator matrix
    j=0
    for i in range(num_new):
        genmat_chunk_list=[] #Generator submatrix for each worker chunk
        for j in range(num_old):
            genmat_chunk_list.append(eyemat*mdsmat[i][j])
        genmat_chunk = np.concatenate(genmat_chunk_list,axis=1)
        genmat_list.append(genmat_chunk)
    genmat = csr_matrix(np.concatenate(genmat_list,axis=0))
    encmat = genmat*mat
    return encmat

def lt_encoder(mat, c, delta, alpha, num_workers):
    #Implement lt encoding as matrix multiplication
    #Generates a single encoded symbol
    rows = mat.shape[0] #80
    encrows = int(alpha*rows) #160
    subrows = int(encrows/num_workers) #20
    num_blocks = rows
    num_enc_blocks = encrows
    u = RS(num_blocks,c,delta)
    genmat= lil_matrix((encrows, rows))
    for enc_block in range(num_enc_blocks):
        # if enc_block%100 == 0:
        #     print (enc_block)
        genmat_block=np.zeros(num_blocks)
        d=1+np.random.choice(num_blocks, size=None, replace=True, p=u) #Degree
        blocks=np.random.choice(num_blocks, size=d, replace=False, p=None) 
        genmat_block[blocks]=1
        genmat[enc_block,:] = genmat_block
    encmat = genmat*mat
    return encmat

def syslt_pardist(mat, c, delta, alpha, num_workers):
    #Returns degrees of parity symbols
    rows = mat.shape[0] 
    sysencrows = int(rows*1.2)
    u = RS(rows,c,delta)
    genmat= lil_matrix((sysencrows, rows))
    deg1_list = []
    genmat_dict = {} #row to col mapping
    genmat_transpose_dict = {} #col to row mapping
    for sysrow in range(sysencrows):
        genmat_row=np.zeros(rows)
        d=1+np.random.choice(rows, size=None, replace=True, p=u) #Degree
        if d==1:
            deg1_list.append(sysrow)
        blocks=np.random.choice(rows, size=d, replace=False, p=None) 
        genmat_dict[sysrow] = blocks.tolist()
        for block in genmat_dict[sysrow]:
            if block in genmat_transpose_dict:
                genmat_transpose_dict[block].append(sysrow)
            else:
                genmat_transpose_dict[block] = [sysrow]
        genmat_row[blocks]=1
        genmat[sysrow,:] = genmat_row
    counter = 0
    src_num = 0
    presrc_src_map = {} #Key = presrc_num, value = src_nums
    # genmat_temp = 0
    deg1_keys = []
    while counter < rows:
        if len(deg1_list)==0:
            print ("ERROR")
            break
        sysrow_c = deg1_list.pop()
        if len(genmat_dict[sysrow_c]) == 0:
            continue
        deg1_keys.append(sysrow_c) #Recording degree 1 symbols discovered until now
        if sysrow_c in presrc_src_map:
            presrc_src_map[sysrow_c].append(src_num)
        else:
            presrc_src_map[sysrow_c] = [src_num]
        # print (counter)
        # print (genmat_dict[sysrow_c])
        col = genmat_dict[sysrow_c][0] #col corresponding to non zero element
        rowslist = genmat_transpose_dict[col] #Rows containing that column
        for row in rowslist:
            #Deleting col from rows in rowslist and adding src_num to list of src symbols in row
            if row in presrc_src_map:
                presrc_src_map[row].append(src_num)
            else:
                presrc_src_map[row] = [src_num]
            genmat_dict[row].remove(col)
            if len(genmat_dict[row])==1:
                #Appending newly created degree 1 rows to deg1_list
                deg1_list.append(row)
        counter+=1
        src_num+=1
    presrc_src_map = {key: presrc_src_map[key] for key in deg1_keys} #check this
    presrc_keys = list(presrc_src_map.keys())
    num_parity = int((alpha-1)*rows) 
    parity_degdist = np.zeros(num_parity)
    for parity_row in range(num_parity):
        genmat_row=np.zeros(rows)
        d=1+np.random.choice(rows, size=None, replace=True, p=u) #Degree
        presrc_rows=np.random.choice(rows, size=d, replace=False, p=None).tolist()
        presrc_subkeys = [presrc_keys[i] for i in presrc_rows]
        presrc_src_submap = {key: presrc_src_map[key] for key in presrc_subkeys}
        # print (presrc_src_submap)
        src_rows_list = get_union(presrc_src_submap)
        # print (src_rows_list)
        enc_vect = np.sum(mat[src_rows_list,:],axis=0)
        # print (enc_vect)
        parity_degdist[parity_row] = np.count_nonzero(enc_vect)
    # print (parity_degdist.shape)
    # print (parity_degdist)
    return parity_degdist