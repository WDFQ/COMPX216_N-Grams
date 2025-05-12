import re
import math
import random

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.
    # Replace the line below with your code.
    
    #create outer dictionary 
    outer_dictionary = {}
    #create inner dictionary
    inner_dictionary = {}

    #loop through the text document
    for word in sequence:
        #if word is already in inner dictionary
        if word in inner_dictionary:
            #increment the value for that key (word)
            inner_dictionary[word] += 1
        else:
            #create a new entry in the inner dictionary and make value = 1
            inner_dictionary[word] = 1
    
    #make value of an entry in the outer dictionary the inner dictionary and make key emtpy tuple "()"
    outer_dictionary[()] = inner_dictionary
    return outer_dictionary
    

def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.
    # Replace the line below with your code.

    #current and previous word
    prev_word = None
    
    #create outer dictionary 
    outer_dictionary = {}
    

    #loop through the text document
    for word in sequence:

        if(prev_word is None):
            #prev word 
            prev_word = (word, )
            continue

        #if prev word is in outer dictionary 
        if(prev_word in outer_dictionary):
            inner_dictionary = outer_dictionary[prev_word]

             #if word is already in inner dictionary
            if word in inner_dictionary:
                #increment the value for that key (word)
                inner_dictionary[word] += 1
            else:
                #create a new entry in the inner dictionary and make value = 1
                inner_dictionary[word] = 1

            #set value of outer dict to inner dict with current word
            outer_dictionary[prev_word] = inner_dictionary
        else:
            #create
            inner_dictionary = {}
            #create a new entry in the inner dictionary and make value = 1
            inner_dictionary[word] = 1
             #set value of outer dict to inner dict with current word
            outer_dictionary[prev_word] = inner_dictionary

        prev_word = (word, )
            
    return outer_dictionary

    

def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    # Replace the line below with your code.

    #gets the first n no of context words 
    prev_words = tuple(sequence[:n - 1])
    
    #create outer dictionary 
    outer_dictionary = {}
    

    #loop through the text document
    for word in sequence:

        #check if the initial length has enough context to be put in to the outer dict
        if word in prev_words:
            continue

        #if prev words is in outer dictionary 
        if(prev_words in outer_dictionary):
            inner_dictionary = outer_dictionary[prev_words]

            #if word is already in inner dictionary
            if word in inner_dictionary:
                #increment the value for that key (word)
                inner_dictionary[word] += 1
            else:
                #create a new entry in the inner dictionary and make value = 1
                inner_dictionary[word] = 1

            #set value of outer dict to inner dict with current words
            outer_dictionary[prev_words] = inner_dictionary
        else:
            #create
            inner_dictionary = {}
            #set value of outer dict to inner dict with current word
            outer_dictionary[prev_words] = inner_dictionary
            #create a new entry in the inner dictionary and make value = 1
            inner_dictionary[word] = 1
            


        #update tuple 
        prev_words_list = []
        counter = 0
        for old_word in prev_words:
            if counter == 0:
                counter += 1
                continue
            else:
                prev_words_list.append(old_word)
            counter += 1

        prev_words_list.append(word)

        prev_words = (tuple)(prev_words_list)

            
    return outer_dictionary
    

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    if sequence in model:
        return model[sequence]
    else:
        return None

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor
    comp = 1 - factor
    for pred in preds[:-1]:
        if pred:
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            mult = comp * factor
            comp -= mult
    pred = preds[-1]
    mult += comp
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.

    inner_dicts = []

    models_used_list = []


    #put all models in to models used list if current sequence length is more than 9
    if len(sequence) > 9:
        models_used_list = models
    #use the models from the first one to the length of sequence + 1 
    else:
        models_used_list = models[-(len(sequence) + 1):]
    

    #counter for how many context words we need at the moment
    counter = len(models_used_list) - 1
    
    for model in models_used_list:
        
        #if we have context words, grab counter amount from the end of the sequence
        if(counter > 0):
            context_words = tuple(sequence[-(counter):])
        else:
            context_words = ()

        
        #grabs the inner dictionary 
        #print(context_words)
        inner = query_n_gram(model, context_words)
        
        #if current inner dict has those context words, add to all the inner dicts list
        if inner is not None:
            inner_dicts.append(inner)
            
        # -1 from counter to use all the other n grams
        counter -= 1

    #gets the blended probabilties and puts into a dictionary
    blended_prob_dict = blended_probabilities(inner_dicts)

    #returns the a random word
    #key = words avalible to choose from, values = weight of each word, 1 = choosing one word
    return random.choices(list(blended_prob_dict.keys()), weights=(list)(blended_prob_dict.values()), k = 1)[0]

                

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError
    #

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    '''
    model = build_unigram(sequence[:20])
    print(model)
    '''

    # Task 1.2 test code
    '''
    model = build_bigram(sequence[:20])
    print(model)
    '''

    # Task 1.3 test code
    model = build_n_gram(sequence[:20], 5)
    print(model)

    # Task 2 test code
    print()
    print(query_n_gram(model, tuple(sequence[:4])))


    # Task 3 test code
    
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    

    # Task 4.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 4.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''
