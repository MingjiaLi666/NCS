from src.optimizers import OpenAIOptimizer, CanonicalESOptimizer, CanonicalESMeanOptimizer, NCSOptimizer
from src.policy import Policy
from src.logger import Logger

from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import gym


# This will allow us to create optimizer based on the string value from the configuration file.
# Add you optimizers to this dictionary.
optimizer_dict = {
    'OpenAIOptimizer': OpenAIOptimizer,
    'CanonicalESOptimizer': CanonicalESOptimizer,
    'CanonicalESMeanOptimizer': CanonicalESMeanOptimizer,
    'NCSOptimizer':NCSOptimizer
}


# Main function that executes training loop.
# Population size is derived from the number of CPUs
# and the number of episodes per CPU.
# One CPU (id: 0) is used to evaluate currently proposed
# solution in each iteration.
# run_name comes useful when the same hyperparameters
# are evaluated multiple times.
def main(ep_per_cpu, game, configuration_file, run_name):
    start_time = time.time()

    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())

    env_name = '%sNoFrameskip-v4' % game

    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    # One cpu (rank 0) will evaluate results
    train_cpus = cpus - 1
    k = 10
    epoch = 5

    # Deduce population size
    lam = train_cpus * ep_per_cpu

    # Create environment
    env = gym.make(env_name)

    # Create policy (Deep Neural Network)
    # Internally it applies preprocessing to the environment state
    policy = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])

    # Create reference batch used for normalization
    # It will be overwritten with vb from worker with rank 0
    vb = policy.get_vb()

    # Extract vector with current parameters.
    parameters = policy.get_parameters()

    # Send parameters from worker 0 to all workers (MPI stuff)
    # to ensure that every worker starts in the same position

    #comm.Bcast([parameters, MPI.FLOAT], root=0)
    comm.Bcast([vb, MPI.FLOAT], root=0)

    # Create optimizer with user defined settings (hyperparameters)
    OptimizerClass = optimizer_dict[configuration['optimizer']]
    optimizer = OptimizerClass(parameters, lam, rank, configuration["settings"])

    # Set the same virtual batch for each worker
    if rank != 0:
        policy.set_vb(vb)
        lens = [0] * ep_per_cpu
        rews = [0] * ep_per_cpu
        # inds = [0] * ep_per_cpu
        nums = [0] * ep_per_cpu
        sig = [0] *ep_per_cpu
        paras = None

        # For each episode in this CPU we get new parameters,
        # update policy network and perform policy rollout
        for i in range(ep_per_cpu):
            e_r = 0
            e_l = 0
            ind, p = optimizer.get_parameters()
            for j in range(k):
                policy.set_parameters(p)
                e_rew, e_len = policy.rollout()
                e_r += e_rew
                e_l += e_len
            lens[i] = e_l
            rews[i] = e_r/k
            # inds[i] = ind
            paras = p
            nums [i] = rank + i
            sig [i] = optimizer.sigma
        # dictionary = {
        #     'parameter':None,
        #     'rank':rank
        #
        # }

        msg = np.array(rews + lens+nums+sig, dtype=np.int32)
        pp = paras.flatten()

    # Only rank 0 worker will log information from the training
    logger = Logger(optimizer.log_path(game, configuration['network'], run_name))
    if rank == 0:
        # Initialize logger, save virtual batch and save some basic stuff at the beginning
        logger.save_vb(vb)

        # Log basic info from the optimizer
        optimizer.log_basic(logger)
        msg = np.zeros(4 * ep_per_cpu, dtype=np.int32)
        pp = np.zeros((ep_per_cpu,optimizer.n))
    results = np.empty((cpus, 4 * ep_per_cpu), dtype=np.int32)
    ppp = np.empty((cpus,optimizer.n*ep_per_cpu))

    comm.Allgather([msg, MPI.INT], [results, MPI.INT])
    comm.Allgather([pp, MPI.FLOAT], [ppp, MPI.FLOAT])
    results = results[1:, :]
    ppp = ppp[1:,:]
    ppp = ppp.flatten()
    rews = results[:, :ep_per_cpu].flatten()
    nums = results[:, (2 * ep_per_cpu):(3 * ep_per_cpu)].flatten()
    sigmas = results[:, (3 * ep_per_cpu):].flatten()
    comm.Bcast([ppp, MPI.FLOAT], root=0)
    comm.Bcast([rews, MPI.FLOAT], root=0)
    comm.Bcast([sigmas, MPI.FLOAT], root=0)
    # comm.Bcast([rews, MPI.FLOAT], root=0)
    # comm.Bcast([ppp, MPI.FLOAT], root=0)
    if rank == 0:
        BestScore = max(rews)
        BestS = {'score':BestScore}
        Bestid = np.argmax(rews)
        BestFound = ppp[Bestid*optimizer.n:(Bestid+1)*optimizer.n]
        logger.log('Best'.ljust(25) + '%f' %BestScore )

    # We will count number of steps
    # frames = 4 * steps (3 * steps for SpaceInvaders)
    steps_passed = 0
    iteration =1


    while steps_passed<=25000000:
        # Iteration start time
        iter_start_time = time.time()
        if iteration % epoch == 1:
            updateCount = [0]*train_cpus
        llambda = np.random.normal(1,0.1-0.1*steps_passed/25000000)
        # Workers that run train episodes
        if rank != 0:
            # Empty arrays for each episode. We save: length, reward, noise index
            lens1 = [0] * ep_per_cpu
            rews1 = [0] * ep_per_cpu
            nums1 = [0] * ep_per_cpu
            sig1 = [0] * ep_per_cpu
            paras1 = None
            

            # For each episode in this CPU we get new parameters,
            # update policy network and perform policy rollout
            for i in range(ep_per_cpu):
                e_r = 0
                e_l = 0
                p = ppp[((rank-1)+i)*optimizer.n:(rank+i)*optimizer.n] + np.random.normal(scale = optimizer.sigma,size = optimizer.n)
                for j in range(k):
                    policy.set_parameters(p)
                    e_rew, e_len = policy.rollout()
                    e_r += e_rew
                    e_l += e_len
                #p = np.random.normal(scale = optimizer.sigma,size = optimizer.n)
                #     BestS['score'] = e_rews
                #     BestFound = p
                #     comm.bcast(BestS, root=rank)
                #     comm.Bcast([BestFound, MPI.FLOAT], root=rank)

                lens1[i] = e_l
                rews1[i] = e_r/k
                nums1[i] = rank+i
                sig1[i] = optimizer.sigma
                paras1 = p

            # Aggregate information, will later send it to each worker using MPI
            msg1 = np.array(rews1 + lens1 + nums1+sig1, dtype=np.int32)
            pp1 = paras1.flatten()

        # Worker rank 0 that runs evaluation episodes
        else:
            #rews1 = [0] * (ep_per_cpu*train_cpus)
            #lens1 = [0] * (ep_per_cpu*train_cpus)
            #for j in range(train_cpus):
                #for i in range(ep_per_cpu):
                    #optimizer.parameters = ppp[(i+j) * optimizer.n:( i+j+1) * optimizer.n]
                    #ind, p = optimizer.get_parameters()
                    #policy.set_parameters(p)
                    #reward = 0
                    #for k in range(10):
                        #e_rew, e_len = policy.rollout()
                        #reward +=e_rew
                    #rews1[i] = reward
                    #lens1[i] = e_len

            # Empty array, evaluation results are not used for the update
            msg1 = np.zeros(4 * ep_per_cpu, dtype=np.int32)
            pp1 = np.zeros((ep_per_cpu, optimizer.n))
        # MPI stuff
        # Initialize array which will be updated with information from all workers using MPI
        results1 = np.empty((cpus, 4 * ep_per_cpu), dtype=np.int32)
        ppp1 = np.empty((cpus, optimizer.n * ep_per_cpu))
        comm.Allgather([msg1, MPI.INT], [results1, MPI.INT])
        comm.Allgather([pp1, MPI.FLOAT], [ppp1, MPI.FLOAT])
        ppp1 = ppp1[1:, :]
        ppp1 = ppp1.flatten()
        

        # Skip empty evaluation results from worker with id 0
        results1 = results1[1:, :]

        # Extract IDs and rewards
        rews1 = results1[:, :ep_per_cpu].flatten()
        lens1 = results1[:, ep_per_cpu:(2*ep_per_cpu)].flatten()
        sigmas1 = results1[:, (3 * ep_per_cpu):].flatten()
        if rank ==0:
            # Update parameters
            # ppp, nums,rews = optimizer.update(ppp, ppp1, nums, nums1, rews, rews1)
            newBestidx = np.argmax(rews1)
            eval_mean_rew = np.mean(rews)
            eval_mean_rew1 = np.mean(rews1)
            if np.max(rews1)>BestS['score']:
                BestS['score'] = rews1[newBestidx]
                BestFound = ppp1[newBestidx*optimizer.n:(newBestidx+1)*optimizer.n]
            Corr = [0] * train_cpus
            Corr1 = [0] * train_cpus

            for i in range(train_cpus):
                para = ppp[optimizer.n*i:optimizer.n*(i+1)]
                para1 = ppp1[optimizer.n * i:optimizer.n * (i + 1)]

                Corr[i] = calCorr(optimizer.n,ppp, para,sigmas,sigmas[i],i)
                Corr1[i] = calCorr(optimizer.n, ppp, para1, sigmas, sigmas1[i], i)
            Corr = np.array(Corr)
            Corr1 = np.array(Corr1)
            rews = np.array(rews)
            rews1 = np.array(rews1)
            Corr1 = Corr1/(Corr+Corr1)
            fx = -rews + BestS['score']
            fx1 = -rews1 + BestS['score']
            # rews = rews/(rews+rews1)
            fx1 = fx1 / (fx + fx1)
            jud = fx1/Corr1
            # print(jud)

            #now update
            for i in range(train_cpus):
                # logger.log("update %s" % i)
                # logger.log(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                if jud[i] <llambda:
                    ppp[optimizer.n*i:optimizer.n*(i+1)] = ppp1[optimizer.n * i:optimizer.n * (i + 1)]
                    sigmas[i] = sigmas1[i]
                    rews[i] = rews1[i]
                    updateCount[i]+=1
            comm.Bcast([updateCount, MPI.FLOAT], root=0)


        # Steps passed = Sum of episode steps from all offsprings
        steps = np.sum(lens1)
        steps_passed += steps

        # Write some logs for this iteration
        # Using logs we are able to recover solution saved
        # after 1 hour of training or after 1 billion frames
        if rank == 0:
            iteration_time = (time.time() - iter_start_time)
            time_elapsed = (time.time() - start_time)/60
            train_mean_rew = np.mean(rews)
            train_max_rew = np.max(rews)
            logger.log('------------------------------------')
            logger.log(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
            logger.log('Iteration'.ljust(25) + '%f' % iteration)
            logger.log('EvalMeanReward'.ljust(25) + '%f' % eval_mean_rew)
            logger.log('EvalMeanReward1'.ljust(25) + '%f' % eval_mean_rew1)
            logger.log('StepsThisIter'.ljust(25) + '%f' % steps)
            logger.log('StepsSinceStart'.ljust(25)+'%f' %steps_passed)
            logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
            logger.log('TimeSinceStart'.ljust(25) + '%d' %time_elapsed)
            logger.log('Best'.ljust(25) + '%f' %BestS['score'])			

            # Give optimizer a chance to log its own stuff
            optimizer.log(logger)
            logger.log('------------------------------------')
            if iteration % 20 == 1:
                fin_rews = 0
                for i in range(30):
                    p = BestFound
                    policy.set_parameters(p)
                    e_rew, e_len = policy.rollout()
                    fin_rews+=e_rew
                fin_eval = fin_rews/30
            else:
                fin_eval = 0     

            # Write stuff for training curve plot
            stat_string = "{},\t{},\t{},\t{},\t{},\t{}\n".\
                format(steps_passed, (time.time()-start_time),
                       eval_mean_rew, eval_mean_rew1, train_mean_rew, fin_eval)
            logger.write_general_stat(stat_string)
            logger.write_optimizer_stat(optimizer.stat_string())

            # Save currently proposed solution every 20 iterations
            if iteration % 20 == 1:
                logger.save_parameters(BestFound, iteration)
        else:
            if iteration%epoch ==0:
                optimizer.updatesigma(epoch,updateCount)
        comm.Bcast([ppp, MPI.FLOAT], root=0)
        comm.Bcast([rews, MPI.FLOAT], root=0)
        comm.Bcast([sigmas, MPI.FLOAT], root=0)
        iteration+=1
    #test best
    if rank == 0:
        final_rews = []
        for i in range(200):
            p = BestFound
            policy.set_parameters(p)
            e_rew, e_len = policy.rollout()
            final_rews.append(e_rew)
        final_eval = np.mean(final_rews)
        logger.log('Final'.ljust(25) + '%f' % final_eval)
        logger.save_parameters(BestFound,iteration)


def calBdistance(n, para,para1,sigma,sigma1):
    xixj = para - para1
    part1 = 1/8*np.dot(xixj,xixj.T)*2/(sigma**2+sigma1**2+1e-8)
    part2 = 1/2*n*np.log(1e-8+(sigma**2+sigma1**2)/(2*sigma*sigma1+1e-8))
    return part1 + part2


def calCorr(n,ppp, para1,sigmalist,sigma1,order1):
    # order = np.argsort(nums)

    DBlist = []
    for i in range(len(sigmalist)):
        if i!=order1:
            para = ppp[n*i:n*(i+1)]

            sigma = sigmalist[i]

            DB = calBdistance(n, para,para1,sigma,sigma1)

            DBlist.append(DB)
    return np.min(DBlist)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes_per_cpu',
                        help="Number of episode evaluations for each CPU, "
                             "population_size = episodes_per_cpu * Number of CPUs",
                        default=1, type=int)
    parser.add_argument('-g', '--game', help="Atari Game used to train an agent")
    parser.add_argument('-c', '--configuration_file', help='Path to configuration file')
    parser.add_argument('-r', '--run_name', help='Name of the run, used to create log folder name', type=str)
    args = parser.parse_args()
    return args.episodes_per_cpu, args.game, args.configuration_file, args.run_name


if __name__ == '__main__':
    ep_per_cpu, game, configuration_file, run_name = parse_arguments()
    main(ep_per_cpu, game, configuration_file, run_name)
