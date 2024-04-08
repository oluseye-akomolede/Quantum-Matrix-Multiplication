from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator
import math
import numpy as np
import random

simulator = AerSimulator()
service = QiskitRuntimeService(channel="ibm_quantum", token='c7f179a3a80319787e05707a688a028d286196ccc238ab7ce187330f9c87b037667525ef06c460c25aad22c48254348e96d479213be1eb880b2ea2d1e9f39e21')
backend = service.least_busy(simulator=False)
random.seed(a = None, version = 2)

def generatenbyn(n):
    result = np.zeros((n,n))
    counter = 1
    for i in range(n):
        for j in range(n):
            result[i][j] = counter/10.0
            counter += 1
    
    return result

def generateidentity(n):
    result = np.identity(n)
    return result


def generatenbynRand(n):
    result = np.zeros((n,n))
    counter = 1
    for i in range(n):
        for j in range(n):
            result[i][j] = random.uniform(-10,10)
            counter += 1
    
    return result


def generate_and_multiply_matrices(NN,nerror):
    #calculates and returns the error for two NNxNN matrices multiplied together using quantum algorithms
    #NN is the size of the square matrix (NNxNN)
    #total_runs is the number of shots for the quantum algorithm
    matrix_A = generatenbynRand(NN)
    matrix_B = generatenbynRand(NN)


    matrix_length = matrix_A.shape[0]
    nmatrix_A = np.zeros((matrix_length,matrix_length))
    nmatrix_B = np.zeros((matrix_length,matrix_length))


    matrix_A_norm = np.zeros(matrix_length)
    matrix_B_norm = np.zeros(matrix_length)

    #calculate the matrix norms and then use the information to determine the number of runs
    for i in range(matrix_length):
        for j in range(matrix_length):
            adderA = math.pow(matrix_A[i][j],2)
            adderB = math.pow(matrix_B[j][i],2)
            matrix_A_norm[i] += adderA
            matrix_B_norm[i] += adderB
        matrix_A_norm[i] = math.sqrt(matrix_A_norm[i]) 
        matrix_B_norm[i] = math.sqrt(matrix_B_norm[i]) 

    tRunA = 0
    tRunB = 0

    for i in range(matrix_length):
        tRunA += math.pow(matrix_A_norm[i],2)
        tRunB += math.pow(matrix_B_norm[i],2)
    
    tRunA = math.sqrt(tRunA)
    tRunB = math.sqrt(tRunB)

    total_runs = math.ceil((tRunA*tRunB)/nerror)
    #total_runs = 1000

    for i in range(matrix_length):
        for j in range(matrix_length):
            nmatrix_A[i][j] = matrix_A[i][j] / matrix_A_norm[i]
            nmatrix_B[j][i] = matrix_B[j][i] / matrix_B_norm[i]


        

    nQbits_vec = math.ceil(math.log2(matrix_length))  #number of qbits
    tot_nQbits = 1 + nQbits_vec
    nQbits_amp = 2**tot_nQbits
    vector_amplitudes = np.zeros(nQbits_amp)   #create the vector amplitude matrix



    matrix_Output = np.zeros((matrix_length,matrix_length))

    for i in range(matrix_length):
        
        for j in range(matrix_length):

            #this could likely be handled by gpus
            for k in range(matrix_length):
                vector_amplitudes[k] = (nmatrix_A[i][k] + nmatrix_B[k][j]) * (1/math.sqrt(4))
                vector_amplitudes[k+2**nQbits_vec] = (nmatrix_A[i][k] - nmatrix_B[k][j]) * (1/math.sqrt(4))
            #end code for gpu replacement
            
            q = QuantumRegister(tot_nQbits)
            c = ClassicalRegister(1)
            qc = QuantumCircuit(q,c)

            #create our list of qbits
            qbit_list = []

            for k in range(tot_nQbits):
                qbit_list.append(q[k])

            qc.prepare_state(vector_amplitudes,qbit_list)       


            qc.measure(q[tot_nQbits-1],c)
            #compiled_circuit = transpile(qc,simulator)

            sampler = Sampler(backend)
            compiled_circuit = transpile(qc,backend)

            job = sampler.run(compiled_circuit, shots = total_runs)
            result = job.result()
            counts = result.get_counts(compiled_circuit)
            try:
                success_counts = counts['1']
            except:
                success_counts = 0
            multiplier_s = success_counts/total_runs
            prob_adjust = 1-2*multiplier_s
            matrix_Output[i][j] = matrix_A_norm[i] * matrix_B_norm[j] * (1-2.0*(multiplier_s))

    matrix_product = matrix_A.dot(matrix_B)
    matrix_diff = matrix_product - matrix_Output
    error = 0
    for i in range(matrix_length):
        temp = 0
        for j in range(matrix_length):
            temp += abs(matrix_diff[i][j])/((abs(matrix_product[i][j])+0.00000001)*math.pow(NN,2))
        error += temp

    #error = np.linalg.norm(matrix_A.dot(matrix_B) - matrix_Output,'fro')/NN

    

    print(error)
    return error


def convertAmp4RepCode(iNDx, n_rep_bits):
    #converts the vector amplitude to something that can be used by a repetition code
    result = 0
    bin_index = "{0:b}".format(iNDx)
    temp = int(bin_index,10)
    temp = temp * ((2**n_rep_bits)-1)
    temp = str(temp)
    result = int(temp,2**n_rep_bits)

    return result

def generate_and_multiply_matrices_RepCode(NN,total_runs):
    #calculates and returns the error for two NNxNN matrices multiplied together using quantum algorithms
    #utilizes repetition codes to try and minimize error
    #NN is the size of the square matrix (NNxNN)
    #total_runs is the number of shots for the quantum algorithm
    matrix_A = generatenbyn(NN)
    matrix_B = generatenbyn(NN)

    
    


    matrix_length = matrix_A.shape[0]
    nmatrix_A = np.zeros((matrix_length,matrix_length))
    nmatrix_B = np.zeros((matrix_length,matrix_length))

    #number of bits for the repetition code
    n_rep_bits = max((3,math.floor(math.log10(matrix_length))))


    matrix_A_norm = np.zeros(matrix_length)
    matrix_B_norm = np.zeros(matrix_length)

    for i in range(matrix_length):
        for j in range(matrix_length):
            adderA = math.pow(matrix_A[i][j],2)
            adderB = math.pow(matrix_B[j][i],2)
            matrix_A_norm[i] += adderA
            matrix_B_norm[i] += adderB
        matrix_A_norm[i] = math.sqrt(matrix_A_norm[i]) 
        matrix_B_norm[i] = math.sqrt(matrix_B_norm[i]) 

    for i in range(matrix_length):
        for j in range(matrix_length):
            nmatrix_A[i][j] = matrix_A[i][j] / matrix_A_norm[i]
            nmatrix_B[j][i] = matrix_B[j][i] / matrix_B_norm[i]


        

    nQbits_vec = math.ceil(math.log2(matrix_length))  #number of qbits
    tot_nQbits = (1 + nQbits_vec) * n_rep_bits
    nQbits_amp = 2**tot_nQbits
    vector_amplitudes = np.zeros(nQbits_amp)   #create the vector amplitude matrix



    matrix_Output = np.zeros((matrix_length,matrix_length))
    matrix_Output0 = np.zeros((matrix_length,matrix_length))

    for i in range(matrix_length):
        
        for j in range(matrix_length):

            vector_amplitudes = np.zeros(nQbits_amp)   #create the vector amplitude matrix
            #this could likely be handled by gpus
            for k in range(matrix_length):
                plus_i = k
                minus_i = k+2**nQbits_vec

                plus_k = convertAmp4RepCode(plus_i,n_rep_bits)
                minus_k = convertAmp4RepCode(minus_i,n_rep_bits)
                vector_amplitudes[plus_k] = (nmatrix_A[i][k] + nmatrix_B[k][j]) * (1/math.sqrt(4))
                vector_amplitudes[minus_k] = (nmatrix_A[i][k] - nmatrix_B[k][j]) * (1/math.sqrt(4))
            #end code for gpu replacement
            
            q = QuantumRegister(tot_nQbits)
            c = ClassicalRegister(n_rep_bits)
            qc = QuantumCircuit(q,c)

            #create our list of qbits
            qbit_list = []

            for k in range(tot_nQbits):
                qbit_list.append(q[k])

            qc.prepare_state(vector_amplitudes,qbit_list)       

            #for k in range(n_rep_bits):
            #    qc.measure(q[tot_nQbits-n_rep_bits + k],c[k])

            qc.measure(q[tot_nQbits-n_rep_bits:tot_nQbits],c)
            qc.draw('mpl')

            compiled_circuit = transpile(qc,simulator)
            job = simulator.run(compiled_circuit, shots = total_runs)
            result = job.result()
            counts = result.get_counts(compiled_circuit)
            try:
                success_counts = 0
                success_counts0 = 0
                success_counts0 = counts['000']
                success_counts = counts['111']
            except:
                success_counts0 = total_runs
                success_counts = 0
            multiplier_s = success_counts/total_runs
            multiplier_s0 = success_counts0/total_runs
            prob_adjust = 1-2*multiplier_s

            
            matrix_Output0[i][j] = matrix_A_norm[i] * matrix_B_norm[j] * (2.0*(multiplier_s0)-1)
            
            matrix_Output[i][j] = matrix_A_norm[i] * matrix_B_norm[j] * (1-2.0*(multiplier_s))

    error = np.linalg.norm(matrix_A.dot(matrix_B) - matrix_Output,'fro')/math.pow(NN,2)
    error0 = np.linalg.norm(matrix_A.dot(matrix_B) - matrix_Output0,'fro')/math.pow(NN,2)
    print(error)
    print(error0)
    return error


def multiply_matrices_square_estimator(matrix_A,matrix_B,n_e):
    nerror = n_e
    matrix_length = matrix_A.shape[0]
    nmatrix_A = np.zeros((matrix_length,matrix_length))
    nmatrix_B = np.zeros((matrix_length,matrix_length))


    matrix_A_norm = np.zeros(matrix_length)
    matrix_B_norm = np.zeros(matrix_length)

    #calculate the matrix norms and then use the information to determine the number of runs
    for i in range(matrix_length):
        for j in range(matrix_length):
            adderA = math.pow(matrix_A[i][j],2)
            adderB = math.pow(matrix_B[j][i],2)
            matrix_A_norm[i] += adderA
            matrix_B_norm[i] += adderB
        matrix_A_norm[i] = math.sqrt(matrix_A_norm[i]) 
        matrix_B_norm[i] = math.sqrt(matrix_B_norm[i]) 

    tRunA = 0
    tRunB = 0

    for i in range(matrix_length):
        tRunA += math.pow(matrix_A_norm[i],2)
        tRunB += math.pow(matrix_B_norm[i],2)
    
    tRunA = math.sqrt(tRunA)
    tRunB = math.sqrt(tRunB)

    total_runs = math.ceil((tRunA*tRunB)/nerror)
    

    for i in range(matrix_length):
        for j in range(matrix_length):
            nmatrix_A[i][j] = matrix_A[i][j] / matrix_A_norm[i]
            nmatrix_B[j][i] = matrix_B[j][i] / matrix_B_norm[i]


        

    nQbits_vec = math.ceil(math.log2(matrix_length))  #number of qbits
    tot_nQbits = 1 + nQbits_vec
    nQbits_amp = 2**tot_nQbits
    vector_amplitudes = np.zeros(nQbits_amp)   #create the vector amplitude matrix



    matrix_Output = np.zeros((matrix_length,matrix_length))

    for i in range(matrix_length):
        
        for j in range(matrix_length):

            #this could likely be handled by gpus
            for k in range(matrix_length):
                vector_amplitudes[k] = (nmatrix_A[i][k] + nmatrix_B[k][j]) * (1/math.sqrt(4))
                vector_amplitudes[k+2**nQbits_vec] = (nmatrix_A[i][k] - nmatrix_B[k][j]) * (1/math.sqrt(4))
            #end code for gpu replacement
            
            q = QuantumRegister(tot_nQbits)
            c = ClassicalRegister(1)
            qc = QuantumCircuit(q,c)

            #create our list of qbits
            qbit_list = []

            for k in range(tot_nQbits):
                qbit_list.append(q[k])

            qc.prepare_state(vector_amplitudes,qbit_list)       


            qc.measure(q[tot_nQbits-1],c)

            estimator = Estimator(backend)
            compiled_circuit = transpile(qc,backend)
            job = estimator.run(compiled_circuit, shots = total_runs)

            result = job.result()
            
            qdists = result.quasi_dists


            try:
                multiplier_s = qdists[0][1]
            except:
                multiplier_s = 0
            matrix_Output[i][j] = matrix_A_norm[i] * matrix_B_norm[j] * (1-2.0*(multiplier_s))

    return matrix_Output


def multiply_matrices_square(matrix_A,matrix_B,n_e):
    nerror = n_e
    matrix_length = matrix_A.shape[0]
    nmatrix_A = np.zeros((matrix_length,matrix_length))
    nmatrix_B = np.zeros((matrix_length,matrix_length))


    matrix_A_norm = np.zeros(matrix_length)
    matrix_B_norm = np.zeros(matrix_length)

    #calculate the matrix norms and then use the information to determine the number of runs
    for i in range(matrix_length):
        for j in range(matrix_length):
            adderA = math.pow(matrix_A[i][j],2)
            adderB = math.pow(matrix_B[j][i],2)
            matrix_A_norm[i] += adderA
            matrix_B_norm[i] += adderB
        matrix_A_norm[i] = math.sqrt(matrix_A_norm[i]) 
        matrix_B_norm[i] = math.sqrt(matrix_B_norm[i]) 

    tRunA = 0
    tRunB = 0

    for i in range(matrix_length):
        tRunA += math.pow(matrix_A_norm[i],2)
        tRunB += math.pow(matrix_B_norm[i],2)
    
    tRunA = math.sqrt(tRunA)
    tRunB = math.sqrt(tRunB)

    total_runs = math.ceil((tRunA*tRunB)/nerror)
    

    for i in range(matrix_length):
        for j in range(matrix_length):
            nmatrix_A[i][j] = matrix_A[i][j] / matrix_A_norm[i]
            nmatrix_B[j][i] = matrix_B[j][i] / matrix_B_norm[i]


        

    nQbits_vec = math.ceil(math.log2(matrix_length))  #number of qbits
    tot_nQbits = 1 + nQbits_vec
    nQbits_amp = 2**tot_nQbits
    vector_amplitudes = np.zeros(nQbits_amp)   #create the vector amplitude matrix



    matrix_Output = np.zeros((matrix_length,matrix_length))

    for i in range(matrix_length):
        
        for j in range(matrix_length):

            #this could likely be handled by gpus
            for k in range(matrix_length):
                vector_amplitudes[k] = (nmatrix_A[i][k] + nmatrix_B[k][j]) * (1/math.sqrt(4))
                vector_amplitudes[k+2**nQbits_vec] = (nmatrix_A[i][k] - nmatrix_B[k][j]) * (1/math.sqrt(4))
            #end code for gpu replacement
            
            q = QuantumRegister(tot_nQbits)
            c = ClassicalRegister(1)
            qc = QuantumCircuit(q,c)

            #create our list of qbits
            qbit_list = []

            for k in range(tot_nQbits):
                qbit_list.append(q[k])

            qc.prepare_state(vector_amplitudes,qbit_list)       


            qc.measure(q[tot_nQbits-1],c)

            sampler = Sampler(backend)
            compiled_circuit = transpile(qc,backend)
            job = sampler.run(compiled_circuit, shots = total_runs)

            result = job.result()
            
            qdists = result.quasi_dists


            try:
                multiplier_s = qdists[0][1]
            except:
                multiplier_s = 0
            matrix_Output[i][j] = matrix_A_norm[i] * matrix_B_norm[j] * (1-2.0*(multiplier_s))

    return matrix_Output


def multiply_matrices_square_simulation(matrix_A,matrix_B,n_e):
    nerror = n_e
    matrix_length = matrix_A.shape[0]
    nmatrix_A = np.zeros((matrix_length,matrix_length))
    nmatrix_B = np.zeros((matrix_length,matrix_length))


    matrix_A_norm = np.zeros(matrix_length)
    matrix_B_norm = np.zeros(matrix_length)

    #calculate the matrix norms and then use the information to determine the number of runs
    for i in range(matrix_length):
        for j in range(matrix_length):
            adderA = math.pow(matrix_A[i][j],2)
            adderB = math.pow(matrix_B[j][i],2)
            matrix_A_norm[i] += adderA
            matrix_B_norm[i] += adderB
        matrix_A_norm[i] = math.sqrt(matrix_A_norm[i]) 
        matrix_B_norm[i] = math.sqrt(matrix_B_norm[i]) 

    tRunA = 0
    tRunB = 0

    for i in range(matrix_length):
        tRunA += math.pow(matrix_A_norm[i],2)
        tRunB += math.pow(matrix_B_norm[i],2)
    
    tRunA = math.sqrt(tRunA)
    tRunB = math.sqrt(tRunB)

    total_runs = math.ceil((tRunA*tRunB)/nerror)
    

    for i in range(matrix_length):
        for j in range(matrix_length):
            nmatrix_A[i][j] = matrix_A[i][j] / matrix_A_norm[i]
            nmatrix_B[j][i] = matrix_B[j][i] / matrix_B_norm[i]


        

    nQbits_vec = math.ceil(math.log2(matrix_length))  #number of qbits
    tot_nQbits = 1 + nQbits_vec
    nQbits_amp = 2**tot_nQbits
    vector_amplitudes = np.zeros(nQbits_amp)   #create the vector amplitude matrix



    matrix_Output = np.zeros((matrix_length,matrix_length))

    for i in range(matrix_length):
        
        for j in range(matrix_length):

            #this could likely be handled by gpus
            for k in range(matrix_length):
                vector_amplitudes[k] = (nmatrix_A[i][k] + nmatrix_B[k][j]) * (1/math.sqrt(4))
                vector_amplitudes[k+2**nQbits_vec] = (nmatrix_A[i][k] - nmatrix_B[k][j]) * (1/math.sqrt(4))
            #end code for gpu replacement
            
            q = QuantumRegister(tot_nQbits)
            c = ClassicalRegister(1)
            qc = QuantumCircuit(q,c)

            #create our list of qbits
            qbit_list = []

            for k in range(tot_nQbits):
                qbit_list.append(q[k])

            qc.prepare_state(vector_amplitudes,qbit_list)       


            qc.measure(q[tot_nQbits-1],c)

            compiled_circuit = transpile(qc,simulator)
            job = simulator.run(compiled_circuit, shots = total_runs)
            result = job.result()
            counts = result.get_counts(compiled_circuit)

            
            #qdists = result.quasi_dists


            try:
                success_counts = counts['1']
            except:
                success_counts = 0
            multiplier_s = success_counts/total_runs
            matrix_Output[i][j] = matrix_A_norm[i] * matrix_B_norm[j] * (1-2.0*(multiplier_s))

    return matrix_Output

NNd = 3
mA = generatenbynRand(NNd)
mB = generatenbynRand(NNd)

nerr = 0.001
mO = multiply_matrices_square(mA,mB,nerr)
mO2 = multiply_matrices_square_simulation(mA,mB,nerr)
mO3 = multiply_matrices_square_estimator(mA,mB,nerr)

mStandard = mA.dot(mB)
error = 0
error0 = 0
error1 = 0

for i in range(NNd):
    for j in range(NNd):
        error += abs(mO[i][j]-mStandard[i][j])
        error0 += abs(mO2[i][j] - mStandard[i][j])
        error1 += abs(mO3[i][j]-mStandard[i][j])

error = error/math.pow(NNd,2)
error0 = error0/math.pow(NNd,2)
error1 = error1/math.pow(NNd,2)
#error = np.linalg.norm(mA.dot(mB) - mO,'fro')/math.pow(3,2)
#error0 = np.linalg.norm(mA.dot(mB) - mO2,'fro')/math.pow(3,2)

print(mO)
print(mO2)
print(mO3)
print(mStandard)
print(error)
print(error0)
print(error1)




