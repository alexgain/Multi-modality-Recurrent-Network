def create_breedsets(gen = None, rng = [0.0,1.0], N = 100, p = 10, vec_steps = 4, subset_size = 4, replace = False):
    
    # gen: generator vector set (default = None)
    # rng: range of values in p-D vector (default = [0.0,1.0])
    # N: number of vectors in generator (default = 10)
    # p: dimentionality of the vector (default = 10)
    # vec_steps: factor of size of breed set to generate w.r.t. initializer generator size (default = 4)
    # subset_size: subset size to generate new (default = 4)
    # replace: whether to create breed sets with replacement or not  (default = False)

    num_vectors = vec_steps*N # size of vector subsets to create children

    # create initializer generator of size N with p-D vectors containing values in rng
    if gen is None:
        generator = np.random.uniform(rng[0],rng[1],N*p).reshape((N,p))
    else:
        generator = gen
    
    if(num_vectors*subset_size > N and replace == False):
        #print("Cannot sample where sample space is less than number of samples. Sampling with replacement.")
        replace = True
    
    # create subsets to breed offspring
    breed_set_indices = np.random.choice(range(N),num_vectors*subset_size,replace).reshape((num_vectors,subset_size))

    # select subsets from generator
    breed_sets = list()
    for subset in breed_set_indices:
        breed_sets.append(generator[subset])
        
    return breed_sets

def breednewsets(breed_sets):
    
    # breed_sets: list of breed sets
    
    # breeding function/can be modified (default = average)
    breeder = lambda x: np.mean(x, axis=0)
    
    # breeding the sets to create proginy
    bred = np.empty((0,breed_sets[0].shape[1]), int)
    for subset in breed_sets:
        bred = np.append(bred, np.array([breeder(subset)]), axis=0)
    
    return bred

def fitfun(bred, N = 100):
    
    # bred: prospective set of proginy for generator
    # N: number of elements in generator (default = 10)
    
    # function that arranges proginy by fitness/can be modified (default = maximize L2 norm)
    chosen = -1*np.hstack((bred,np.array([np.apply_along_axis(np.linalg.norm,1,bred)]).T))
    chosen = chosen[chosen[:,0].argsort()]
    chosen = -1*chosen
    chosen = chosen[0:N,0:-1]
    
    return chosen


nextgen = fitfun(breednewsets(create_breedsets()))

for _ in range(100):
    newgen = create_breedsets(nextgen)
    newbreedset = breednewsets(newgen)
    nextgen = fitfun(newbreedset)
