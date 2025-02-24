from multiprocessing import Process, Queue
global __DONE_NUMBER, __NEW_COL_THRSHLDs, __LOCK_1, __KNOWN_DICT_CNT, __KNOWN_DICT


def parallelize_stuff(list_input, method, simultaneous_processes = 10):
    '''
    The smarter way to loop - 
    list_input is a list of list of input arguments [[job1_arg1, job1_arg2, ...], [job2_arg1, job2_arg2, ...], [job3_arg1, job3_arg2, ...]] 
    and method is the method to compute from the input arguments
    the result is a list of output arguments from the given method [jobK_res, jobL_res, ...]

    Here: Lose order of in to output
    '''
    # Initialize
    process_number = len(list_input)
    split_number = simultaneous_processes # split by groups of 10
    task_queue = Queue()
    done_queue = Queue()
    
    cur_runs = 0

    # Submit tasks jedes Put hat: (methode, argumente_tuple) z.B. (multiply, (i, 7))
    for list_in in list_input:
        task_queue.put((method,  list_in))
    
    # Start worker processes
    jobs =[]        
    # Split tasks by defined number
    for i in range(process_number):
        print("Starting task "+str(i+1))
        p = Process(target=_worker, args=(task_queue, done_queue))
        jobs.append(p)

    # Get and print results
    output_list = []
    j = 0
    for i in range(len(jobs)):
        if cur_runs < split_number:
            print("Start job: "+str(i+1))
            jobs[i].start()
            cur_runs +=1
            if len(jobs) != split_number and (len(jobs) - i - 1) < split_number:# remaining_jobs = len(jobs) - i - 1
                j += 1
                print("Received results "+str(j) + " | " + str(len(list_input)))
                output_list.append(done_queue.get())
                #print("Got: "+ str(done_queue.get().head(1)))
            if len(jobs) == split_number and (i +1) == split_number:# remaining_jobs = len(jobs) - i - 1
                j += 1
                print("Received results "+str(j) + " | " + str(len(list_input)))
                output_list.append(done_queue.get())
                #print("Got: "+ str(done_queue.get().head(1)))     
            
        else:
            j += 1
            print("Received results "+str(j) + " | " + str(len(list_input)))
            output_list.append(done_queue.get())

    while j != len(list_input):        
        res = done_queue.get()                
        j += 1
        print("Received results "+str(j) + " | " + str(len(list_input)))
        output_list.append(res)

    # End all 
    for i in range(process_number):
        task_queue.put('STOP')
    
    for job in jobs:
        try:
            job.shutdown()
        except: 
            pass
    return output_list

# ------------- parallelization helpers -------------
    
def _worker(input, output): # Function run by worker processes
    for func, args in iter(input.get, 'STOP'):
        result = _calculate(func, args)
        output.put(result)
        
def _calculate(func, args): # Function used to calculate result
    result = func(*args)
    return result
       
       
       
       
        