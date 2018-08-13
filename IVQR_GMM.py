'''
This is a package for Instrumental Quantile Regression (IVQR) using Generalized Methods of Moments (GMM) estimation.
It is based on the paper: Chen, Le-Yu and Lee, Sokbae (September 2017), 
"Exact computation of GMM estimators for instrumental variable quantile regression models".
The paper has been published at Journal of Applied Econometrics. See https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2619.
The Python codes are based on the MATLAB codes available at:
https://github.com/LeyuChen/IVQR-GMM-computation-codes
This package will use numpy to deal with matrices.
All column vectors are written as 1 by n matrices
'''
import numpy as np
from numpy import concatenate as concat
from numpy.linalg import inv, lstsq
from numpy import diag, zeros, ones, eye
from math import sqrt
from scipy.stats import norm
from gurobipy import *
from scipy.sparse import csr_matrix

### Get Quadratic expression: b_vec_t @ Q_matrix @ b_vec + obj_vec.dot(b_vec) + objcon_vec ###
def get_QExpr(b_vec_t, Q_matrix, b_vec, obj_vec=np.array([0]), objcon_vec=np.array([0])):
    l = b_vec.shape[0]
    objective = QuadExpr()
    for i in range(l):
        for j in range(l):
            objective += b_vec_t[0,i] * Q_matrix[i,j] * b_vec[j,0]
    if np.any(obj_vec):
        objective.add(LinExpr(obj_vec.T[0].tolist(), b_vec.T[0].tolist()))
    objective.addConstant(np.asscalar(objcon_vec))
    return objective
### Get Linear expression: b_vec_t @ obj_vec + objcon_vec ###
def get_LExpr(b_vec, obj_vec=np.array([0]), objcon_vec=np.array([0])):
    objective = LinExpr(obj_vec.T[0].tolist(), b_vec.T[0].tolist())
    objective.addConstant(np.asscalar(objcon_vec))
    return objective
    
### 2SLS function ###
def Two_Stage_LS(y, x, z, robust=1):
    '''
    This function computes the coefficient estimates and the estimated asymptotic variance
    for the two-stage least square (2SLS) regression of y on x using z as instrument
    
    function input :
    y     : the outcome column vector as a NumPy array
    x     : (n by k) matrix of covariate data where k is the number of covariates   
    z     : (n by p) matrix of instrument data where p is the number of instruments
    robust: set robust = 1 for calculating the estimated heteroskedasticity robust asymptotic variance
    
    function output :
    bhat  : the vector of 2SLS regression coefficient estimates
    avar  : the estimated 2SLS asymptotic variance 
    '''
    n = y.shape[0]
    k = x.shape[1]
    z_t = z.T
    P = z @ (lstsq((z_t @ z), z_t)[0])
    xhat = P @ x
    xhat_t = xhat.T
    bhat = lstsq((xhat_t @ xhat), (xhat_t @ y))[0]
    uhat = y - x @ bhat
    uhat_t = uhat.T
    inv_xhatt_xhat = lstsq((xhat_t @ xhat), eye(k))[0]
    if robust == 1:
        u_square = (uhat * uhat).T[0] # since one more dimension is added for transpose
        avar= inv_xhatt_xhat @ ((xhat_t @ diag(u_square) @ xhat) @ inv_xhatt_xhat)
    else:
        avar= ((uhat_t @ uhat) / n) * inv_xhatt_xhat # need to check here the *
    return bhat, avar

### miobnd_fn: calculate the boundary ###
def miobnd_fn(y, x, bnd):
    '''
    Given (y,x), this function solves the following maximization problem
    for each i, max |y(i)-x(i,:)*b| over b confined to the space
    described by bnd

    function input :
    y     : (n by 1) matrix of outcomes
    x     : (n by k) matrix of covariate data    
    bnd   : (k by 2) matrix where the first and second columns  
            respectively store the lower and upper bounds 
            of the unknown coefficients

    function output :
    value : the value of the maximized objective function
    '''
    n = y.shape[0]
    k = x.shape[1]
    # two models here due to objective update problems
    m1 = Model('miobnd_fn1')
    m2 = Model('miobnd_fn2')
    lb=bnd[:,0].tolist()
    ub=bnd[:,1].tolist()
    b1vars = m1.addVars(tuplelist(range(k)), lb=lb, ub=ub, vtype=GRB.CONTINUOUS)
    b1 = np.array(b1vars.select())
    b2vars = m2.addVars(tuplelist(range(k)), lb=lb, ub=ub, vtype=GRB.CONTINUOUS)
    b2 = np.array(b2vars.select())
    m1.update()
    m2.update()
    tol=1e-6
    
    m1.Params.outputflag = 0
    m1.Params.OptimalityTol = tol
    m1.Params.FeasibilityTol = tol
    m1.Params.IntFeasTol = tol
    
    m2.Params.outputflag = 0
    m2.Params.OptimalityTol = tol
    m2.Params.FeasibilityTol = tol
    m2.Params.IntFeasTol = tol

    value=zeros((n,1))
    
    for i in range (n):
        v=zeros((2,1))
        linexpr1 = LinExpr(-x[i,:], b1)
        m1.setObjective(linexpr1+y[i], GRB.MAXIMIZE)
        try:
            A = -x[i,:]
            m1.addLConstr(LinExpr(A, b1) >= -y[i])
            m1.update()
            m1.optimize()
            v[0] = m1.objVal
            m1.remove(m1.getConstrs()[0])
        except GurobiError:
            print('Error reported\n')
            print(v[0])
        linexpr2 = LinExpr(x[i,:], b2)
        m2.setObjective(linexpr2-y[i], GRB.MAXIMIZE)
        try:
            A = x[i,:]
            m2.addLConstr(LinExpr(A, b2) >= y[i])
            m2.update()
            m2.optimize()
            v[1] = m2.objVal
            m2.remove(m2.getConstrs()[0])
        except GurobiError:
            print('Error reported\n')
            print(v[1])
        value[i]=np.amax(v)
    return value

### Mixed integer optimization ###
def IVQR_MIO(y, x, Q, tau, T, abgap, bnd, method):
    '''
    function input :
    y     : vector of outcomes
    x     : (n by k) matrix of the covariate data
    Q     : (n by n) matrix equal to (G*Q_hat*G') stated in the MIQP formulation 
    tau   : quantile index
    T     : the time limit specified for early termination of the MIO solver
    abgap : the absolute gap specified for early termination of the MIO solver
    bnd   : (k by 2) matrix where the first and second columns  
            respectively store the lower and upper bounds 
            of the unknown coefficients
    method: set method=1 for solving the MIQP formulation (3.3)
            set method=2 for solving the MIQP formulation (C.1)
            set method=3 for solving the MILP formulation (C.10)
    
    function output :
    bhat  : the vector of the coefficient estimates
    obj_v : the value of the GMM objective function
    gap   : the MIO optimization gap value in case of early termination
            gap = 0 ==> optimal solution is found
    rtime : the time used by the MIO solver in the estimation procedure
    ncount: the number of nodes already explored by the MIO solver 
    '''
    n = y.shape[0]
    k = x.shape[1]
    bhat = zeros((k, 1))
    
    gap = 0
    rtime = 0
    ncount = 0
    
    tau_vec = ones((n,1)) * tau
    tau_vec_t = tau_vec.T
    
    ##Beginning of the Gurobi Model##
    m = Model(name = "MIQP")
    objcon_vec = tau_vec_t @ Q @ tau_vec
    tol=1e-6

    if method == 1:
        # Use MIQP formulation (3.3)
        print ('solving MIQP formulation (3.3)')
        lb1 = [0]*n
        lb2 = bnd[:,0].tolist()
        ub1 = [1]*n
        ub2 = bnd[:,1].tolist()
        b1 = m.addVars(tuplelist(range(n)), lb=lb1, ub=ub1, vtype=GRB.BINARY)
        b2 = m.addVars(tuplelist(range(n,n+k)), lb=lb2, ub=ub2, vtype=GRB.CONTINUOUS)
        b_vec = concat((np.array([b1.select()]).T, np.array([b2.select()]).T))
        # b_vec is the decision vector
        m.update()
        b_vec_t = b_vec.T
        miobnd = miobnd_fn (y,x,bnd)
        miobnd_bar = miobnd + tol
        rhs_vec = concat((miobnd * (1 - tol) - y, y - tol * miobnd_bar))
        obj_vec = concat((-2 * Q @ tau_vec, zeros((k,1))))
        Q_matrix = csr_matrix(concat((concat((Q, zeros((n,k))), axis=1) ,zeros((k,n+k))))) 
        # the Q_matrix in the model, not the input Q
        A_matrix = concat((concat((diag(miobnd.T[0]), -x), axis=1), concat((-diag(miobnd_bar.T[0]), x), axis=1)))
        objective = get_QExpr(b_vec_t, Q_matrix, b_vec, obj_vec, objcon_vec)
        m.setObjective(objective, GRB.MINIMIZE)
        ### Two constraints adding methods, decide by speed ###
#        m.addConstrs(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) <= rhs_vec[i,0] for i in range (n+k))
        ### Alternatively ###
        for i in range(n+k):
            m.addLConstr(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) <= rhs_vec[i,0])

    elif method == 2:
        # MIQP formulation (C.1)
        print ('solving MIQP formulation (C.1)')
        eps = np.finfo(float).eps
        lb1 = [0] * n
        lb2 = bnd[:,0].tolist()
        lb3 = [0] * n
        lb4 = [0] * (2*n)
        ub1 = [1] * n
        ub2 = bnd[:,1].tolist()
        ub3 = [1] * n
        ub4 = [1/eps] * (2*n)
        b1 = m.addVars(tuplelist(range(n)), lb=lb1, ub=ub1, vtype=GRB.BINARY)
        b2 = m.addVars(tuplelist(range(n,n+k)), lb=lb2, ub=ub2, vtype=GRB.CONTINUOUS)
        b3 = m.addVars(tuplelist(range(n+k, 2*n+k)), lb=lb3, ub=ub3, vtype=GRB.BINARY)
        b4 = m.addVars(tuplelist(range(2*n+k,4*n+k)), lb=lb4, ub=ub4, vtype=GRB.CONTINUOUS)
        m.update()
        b_vec = concat((np.array([b1.select()]).T, np.array([b2.select()]).T, \
                        np.array([b3.select()]).T, np.array([b4.select()]).T))
        b_vec_t = b_vec.T
        rhs_vec = concat((ones((n,1)), y, (-1e-5)*ones((n,1))))
        obj_vec = concat((2 * Q @ tau_vec, zeros((3*n+k,1))))
        Q_matrix = csr_matrix(concat((concat((Q, zeros((n,3*n+k))), axis=1), zeros((3*n+k,4*n+k)))))
        A_matrix = concat((concat((eye(n), zeros((n,k)), eye(n), zeros((n,2*n))), axis=1),\
                                      concat((zeros((n,n)), x, zeros((n,n)), eye(n), -eye(n)), axis=1),\
                                      concat((zeros((n, 2*n+k)), -eye(n), eye(n)), axis=1)))
        objective = get_QExpr(b_vec_t, Q_matrix, b_vec, obj_vec, objcon_vec)
        m.setObjective(objective, GRB.MINIMIZE)
        ### Two constraints adding methods, decide by speed ###
        # m.addConstrs(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) == rhs_vec[i,0] for i in range (2*n))
        # m.addConstrs(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) <= rhs_vec[i,0] for i in range (2*n,3*n))
        ### Alternatively ###
        for i in range (2*n):
            m.addLConstr(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) == rhs_vec[i,0])
        for i in range (2*n,3*n):
            m.addLConstr(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) <= rhs_vec[i,0])
        
        m.Params.PreSOS1BigM = 0

        b_vec_list = b_vec_t[0].tolist()
        for j in range(n):
            m.addSOS(GRB.SOS_TYPE1, [b_vec_list[j], b_vec_list[2*n+k+j]])
            m.addSOS(GRB.SOS_TYPE1, [b_vec_list[n+k+j], b_vec_list[3*n+k+j]])
            
    elif method == 3:
        # MILP formulation (C.10)
        print ('solving MILP formulation (C.10)')
        aux_num = int(n*(n-1)/2)
        lb1 = [0] * n
        lb2 = bnd[:,0].tolist()
        lb3 = [0] * aux_num
        ub1 = [1] * n
        ub2 = bnd[:,1].tolist()
        ub3 = [1] * aux_num   
        b1 = m.addVars(tuplelist(range(n)), lb=lb1, ub=ub1, vtype=GRB.BINARY)
        b2 = m.addVars(tuplelist(range(n,n+k)), lb=lb2, ub=ub2, vtype=GRB.CONTINUOUS)
        b3 = m.addVars(tuplelist(range(n+k, 2*n+k)), lb=lb3, ub=ub3, vtype=GRB.BINARY)
        m.update()
        b_vec = concat((np.array([b1.select()]).T, np.array([b2.select()]).T, np.array([b3.select()]).T))
        b_vec_t = b_vec.T
        miobnd = miobnd_fn(y,x,bnd)  # miobnd_fn computes the values M(i) defined in (3.6)
        miobnd_bar = miobnd + tol
 
        aux_constr1 = zeros((aux_num,n+k+aux_num))
        aux_constr2 = zeros((aux_num,n+k+aux_num))
        aux_constr3 = zeros((aux_num,n+k+aux_num))
        
        s = 0
        for i in range (n-1):

            if i == 0:
                Q_vecl = Q[i+1:n,i:i+1]
            else:
                Q_vecl=concat((Q_vecl, Q[i+1:n,i:i+1]))
            
            # -e_i+x_ij <= 0
            aux_constr1[s:s+n-i-1,i:i+1] = -ones((n-i-1,1))
            aux_constr1[s:s+n-i-1,n+k+s:n+k+s+n-i-1] = eye(n-i-1)

            # -e_j+x_ij <= 0
            aux_constr2[s:s+n-i-1,i+1:n] = -eye(n-i-1)
            aux_constr2[s:s+n-i-1,n+k+s:n+k+s+n-i-1] = eye(n-i-1)
            
            # e_i+e_j-x_ij <= 1
            aux_constr3[s:s+n-i-1,i:i+1] = ones((n-i-1,1))
            aux_constr3[s:s+n-i-1,i+1:n] = eye(n-i-1)
            aux_constr3[s:s+n-i-1,n+k+s:n+k+s+n-i-1] = -eye(n-i-1)

            s += n-i-1

        rhs_vec = concat((miobnd*(1-tol) - y, y - tol*miobnd_bar, zeros((2*aux_num,1)), ones((aux_num,1))))
        obj_vec = concat((diag(Q).reshape(-1,1) - 2*Q @ tau_vec, zeros((k,1)), 2*Q_vecl))
        A_matrix = concat((concat((diag(miobnd.T[0]), -x, zeros((n,aux_num))), axis=1),\
                           concat((-diag(miobnd_bar.T[0]), x, zeros((n,aux_num))), axis=1),\
                           aux_constr1, aux_constr2, aux_constr3))
        objective = get_LExpr(b_vec, obj_vec, objcon_vec)
        m.setObjective(objective, GRB.MINIMIZE)
        
        ### Two constraints adding methods, decide by speed ###
    #    m.addConstrs(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) <= rhs_vec[i,0] for i in range (n+k+aux_num))
        ### Alternatively ###
        for i in range (n+k+aux_num):
            m.addLConstr(LinExpr(A_matrix[i,:].tolist(), b_vec_t.tolist()[0]) <= rhs_vec[i,0])
            
    else:
        print ('error in input arguments')
        return

    m.Params.outputflag = 0
    m.Params.OptimalityTol=tol
    m.Params.FeasibilityTol=tol
    m.Params.IntFeasTol=tol
    if T > 0:
        m.Params.TimeLimit = T
    if abgap > 0:
        m.Params.MIPGapAbs = abgap

    try:
        m.update()
        m.optimize()
        bhat = np.array([[b.x for b in m.getVars()[n:n+k]]]).T
        obj_v = m.objVal
        gap = (obj_v - m.objbound)
        rtime = m.runtime
        ncount = m.nodecount
        status = m.status
        print('Optimization returned status: %s\n' % status)
       
    except (GurobiError, AttributeError):
        print('Error reported\n')
        return
    return bhat, obj_v, gap, rtime, ncount

### Main program for IVQR_GMM calculation ###
def IVQR_GMM(y, w, z, tau, intercept=False, T=0, abgap=0, bnd = np.array([]), method=1): 
    '''
    The IVQR_GMM function computes the exact GMM estimator of the IVQR model
    via the MIO approach as described in Chen and Lee (2017).
    
    function input :
    y        : vector of outcomes
    w        : (n by k) matrix of the covariate dataset, include exogeneous variables
    z        : (n by p ) matrix of the instrument variable dataset, include exogeneous variables
    tau      : quantile index
    intercept: False ==> The function will NOT add intercept term automatically, include it in w and z if needed
               True  ==> The function will ADD intercept term to w and z automatically
    T        : scalar. If T>0, then T is the time limit specified for early termination
               of the MIO solver. Otherwise, the MIO solver keeps running until convergence.
    abgap    : the absolute gap specified for early termination of the MIO solver
    bnd      : (k by 2) matrix where the first and second columns  
               respectively store the lower and upper bounds 
               of the unknown coefficients         
    method   : 1 set method = 1 for computing the GMM estimator based on the formulation (3.3) of Chen and Lee (2017)
               2 set method = 2 for using the formulation (C.1)
               3 set method = 3 for using the formulation (C.10)
    
    The arguments T, abgap and bnd are optional. When they are not specified,
    the following default values are used.
    intercept: set intercept=False, the function will NOT add intercept term by itself
    T        : set T = 0  ==> solve the MIO problem until convergence
    abgap    : set abgap = 0  ==> solve the MIO problem until convergence
    bnd      : Calculate the parameter bounds based on the two-stage least square
               regression results as used in Chen and Lee (2017)
    
    function output :
    theta_hat: the vector of the coefficient estimates
    s_hat    : the estimated asymptotic standard errors
    obj_v    : the value of the GMM objective function
    gap      : the MIO optimization gap value in case of early termination
               gap = 0 ==> optimal solution is found within the time limit
    rtime    : the time used by the MIO solver in the estimation procedure
    ncount   : the number of nodes already explored by the MIO solver 
    '''
    
    n = y.shape[0]
    
    # Add intercept term if needed
    if intercept:
        w = concat((ones((n,1)), w), axis=1)
        z = concat((ones((n,1)), z), axis=1)
    
    # Use of the Hall-Sheath bandwidth choice (Koenker 1994)
    q_tau = norm.ppf(tau)
    H_S_ratio = (norm.pdf(q_tau) ** 2) / (1 + 2 * q_tau * q_tau)
    h_Hall_Sheath =(1.96 ** (2/3)) * ((1.5 / n * H_S_ratio) ** (1/3))
    
    z_t = z.T
    Q = z @ inv(z_t @ z / n) @ (z_t)/(tau * (1 - tau))
    # Q is the matrix G*Q_hat*G' stated in the MIQP formulation of the GMM
    
    # estimation problem
    k = w.shape[1]
    theta_hat = zeros((k, 1))
    
    # calculate boundary by 2SLS if not specified
    if not bnd.size > 0:
        [b, var] = Two_Stage_LS(y, w, z, 1)
        bnd = concat((b - 10*np.sqrt([diag(var)]).T, b + 10*np.sqrt([diag(var)]).T), axis=1)

    [theta_hat, obj_v, gap, rtime, ncount] = IVQR_MIO(y, w, Q, tau, T, abgap, bnd, method)
    
    # compute the estimated asymptotic standard errors based on the method of
    # Powell (1986) using Gaussian kernal and the Hall-Sheath bandwidth choice
    e_hat = y - w @ theta_hat
    kern = norm.pdf(e_hat / h_Hall_Sheath) / h_Hall_Sheath
    k_x = np.matlib.repmat(kern,1,k) * w
    s_hat = np.sqrt(diag(inv(k_x.T @ z @ inv(z_t @ z) @ z_t @ k_x) * tau * (1-tau)))
    return theta_hat, s_hat, obj_v, gap, rtime, ncount

if __name__ == "__main__":
    y = np.array([[0,0,1,0]]).T
    w = np.array([[1,1,2,1],[3,2,3,3]]).T
    z = np.array([[1,1,1,2]]).T
    Q = eye(4)
    intercept = False
    T = 15000
    abgap = 1e-2
    tau = 0.25
    
#    print(IVQR_GMM(y, w, z, tau)
    print(IVQR_MIO(y, w, Q, tau, T, abgap, bnd=np.array([[-20, 20], [-20, 20]]), method=1))
#    print(Two_Stage_LS)
