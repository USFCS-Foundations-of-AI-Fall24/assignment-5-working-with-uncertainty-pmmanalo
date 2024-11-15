import numpy as np
import random
import argparse
import codecs
import os
import numpy

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
                with open(basename + '.trans', 'r') as trans_file:
                    for line in trans_file:
                        if line.startswith('#') or line.strip() is False:
                            continue  
                        a, b, c = line.strip().split()
                        if a not in self.transitions:
                            self.transitions[a] = {}
                        self.transitions[a][b] = float(c)

                with open(basename + '.emit', 'r') as emit_file:
                    for line in emit_file:
                        if line.startswith('#') or line.strip() is False:
                            continue  
                        a, b, c = line.strip().split()
                        if a not in self.emissions:
                            self.emissions[a] = {}
                        self.emissions[a][b] = float(c)


   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        init = random.choice(list(self.transitions.keys()))
        states = []
        outputs = []
        for _ in range(n):
            states.append(init)
            emission = self.emissions[init]
            output = random.choices(list(emission.keys()), weights=emission.values())[0]
            outputs.append(output)
            transition = self.transitions[init]
            init = random.choices(list(transition.keys()), weights=transition.values())[0]
        return states, outputs

    def forward(self, observation):
        #Initialize
        F = np.zeros((len(self.transitions), len(observation)))
        state_list = list(self.transitions.keys())

        #initialize 
        for i in range(len(self.transitions)):
            state = state_list[i]
            F[i, 0] = (1 / len(self.transitions)) * self.emissions[state].get(observation[0], 0)

        for i2 in range(1, len(observation)):
            for j in range(len(self.transitions)):
                state = state_list[j]
                F[j, i2] = sum(
                    F[i, i2 - 1] * self.transitions[state_list[i]][state] * self.emissions[state].get(observation[i2], 0)
                    for i in range(len(self.transitions))
                )

        return state_list[np.argmax(F[:, -1])], np.max(F[:, -1])

    ## you do this: Implement the Viterbi algorithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.
    #allocate a matrix of s states and n observations.
    def viterbi(self, observation):
        #prob and backpointer matrices
        a = [[0 for _ in range(len(observation))] for _ in range(len(list(self.transitions.keys())))]
        b = [[0 for _ in range(len(observation))] for _ in range(len(list(self.transitions.keys())))]

        #initialize
        for s in range(len(list(self.transitions.keys()))):
            a[s][0] = (1 / len(list(self.transitions.keys()))) * self.emissions[list(self.transitions.keys())[s]].get(observation[0], 0) #probability
            b[s][0] = 0 #backpointer

        for i in range(1, len(observation)):
            for j in range(len(list(self.transitions.keys()))):
                maxprob = 0
                bestprev = 0
                for k in range(len(list(self.transitions.keys()))):
                    current_prob = a[k][i-1] * self.transitions[list(self.transitions.keys())[k]].get(list(self.transitions.keys())[j], 0) * self.emissions[list(self.transitions.keys())[j]].get(observation[i], 0)
                    if current_prob > maxprob:
                        maxprob = current_prob
                        bestprev = k
                a[j][i] = maxprob
                b[j][i] = bestprev

        #backtrace
        path = [0] * len(observation)
        best = 0
        max = 0
        for s in range(len(list(self.transitions.keys()))):
            if a[s][len(observation)-1] > max:
                max = a[s][len(observation)-1]
                best = s
        path[len(observation)-1] = best

        starting=-2 
        ending=-1 
        step=-1
        for i in range(len(observation)+starting, ending, step):
            path[i] = b[path[i+1]][i+1]

        seq = [list(self.transitions.keys())[path[i]] for i in range(len(observation))]
        return seq

if __name__ == "__main__":
    ### demo commands here :)
    # python3 hmm.py partofspeech --forward ambiguous_sents.obs
    # python3 hmm.py partofspeech --viterbi ambiguous_sents.obs
    # python3 hmm.py partofspeech --generate 20
    ###
    parser = argparse.ArgumentParser(description='HMM')
    parser.add_argument('basename', type=str, help='basename')
    parser.add_argument('--generate', type=int, help='Generate a sequence')
    parser.add_argument('--viterbi', type=str, help='Viterbi algorithm')
    parser.add_argument('--forward', type=str, help='Forward algorithm')

    args = parser.parse_args()

    hmm = HMM()
    hmm.load('partofspeech')

    if args.generate:
        states, outputs = hmm.generate(args.generate)
        print("States Sequence:", ' '.join(states))
        print("Outputs Sequence:", ' '.join(outputs))

    if args.viterbi:
        with open(args.viterbi, 'r') as file:
            observations = file.read().strip().split()
        most_likely_state_sequence = hmm.viterbi(observations)
        print("Most likely state sequence:", ' '.join(most_likely_state_sequence))

    if args.forward:
        with open(args.forward, 'r') as file:
            observations = file.read().strip().split()
        most_likely_state, probability = hmm.forward(observations)
        print(f"Most likelystate: '{most_likely_state}' with probability {probability}")

    print("HMM Model Loaded")

    # parser = argparse.ArgumentParser(description='HMM')
    # parser.add_argument('basename', type=str, help='basename of the model files')
    # parser.add_argument('--generate', type=int,)
    # args = parser.parse_args()

    # hmm = HMM()
    # hmm.load('partofspeech')

    # if args.generate:
    #     states, outputs = hmm.generate(args.generate)
    #     print("States Sequence:", ' '.join(states))
    #     print("Outputs Sequence:", ' '.join(outputs))
    # print("Starting HMM load test...")
    # model = HMM()
    # model.load('cat')
    # #print("Transitions loaded:", model.transitions)
    # print("Emissions loaded:", model.emissions)

    # parser = argparse.ArgumentParser(description='Run HMM operations')
    # parser.add_argument('basename', type=str)
    # parser.add_argument('--viterbi', type=str)

    # args = parser.parse_args()

    # hmm = HMM()
    # hmm.load('partofspeech')

    # if args.viterbi:
    #     with open(args.viterbi, 'r') as file:
    #         observations = file.read().strip().split()
    #     most_likely_state_sequence = hmm.viterbi(observations)
    #     print("Most likely state sequence:", ' '.join(most_likely_state_sequence))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run HMM')
#     parser.add_argument('basename', type=str)
#     parser.add_argument('--test', action='store_true')
#     parser.add_argument('--forward', nargs='*')

#     args = parser.parse_args()

#     hmm = HMM()
#     hmm.load(args.basename)

#     if args.forward:
#         observations = args.forward  
#         if observations:
#             probability = hmm.forward(observations)
#             print(f"Probability: {probability}")
#         else:
#             print("No observations provided")