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

    #gets the first n context words
    prev_words = tuple(sequence[:n - 1])
    
    #create outer dictionary 
    outer_dictionary = {}
    

    #loop through the text document
    for word in sequence:

        #skip over the first n words
        if word in prev_words:
            continue

        #check if our ngram has the given context words 
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
            


        #update the context words
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
    
    for model in models:  
        #find this model's context length
        context_length = 0
        for key in model.keys():
                #get the legnth of the context of the current model (n gram)
                context_length = len(key) 
                break  
            
        #takes the lower value of either sequence legnth or ngram context length, this decides which ngram to use
        available_context = min(context_length, len(sequence))
        
        #get the context words 
        if available_context > 0:
            context = tuple(sequence[-available_context:]) 
        else:
           context = ()
        
        #get the model's prediction (current inner dictionary) for this context
        inner_dict = query_n_gram(model, context)
        if inner_dict is not None:
            inner_dicts.append(inner_dict)

    #blend all predictions
    blended_probs = blended_probabilities(inner_dicts)
    
    #randomly picks a value based on their weighting 
    words = list(blended_probs.keys())
    weights = list(blended_probs.values())
    return random.choices(words, weights=weights, k=1)[0]
                

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.

    summed_log_values = 0.0

    for i in range(len(sequence)):
        
        if i == 0:
            current_model = models[-1]
            previous_words = ()
        else:
            if i > 9:
                current_model = models[0]
                previous_words = sequence[(i - 9) : i]
            else:
                current_model = models[-(i + 1)]
                previous_words = sequence[:i]
            
        current_word = sequence[i]
            
        #gets inner dict in current model of the given sequence
        inner_dict = query_n_gram(current_model, tuple(previous_words))

        if inner_dict is not None:
            if current_word in inner_dict:
                #getting all the values for all the words in inner dict
                values_list = list(inner_dict.values())
                
                #sum all values in inner dict
                total_value = 0
                for value in values_list:
                    total_value += value
            
                #get probability of current word appearing in inner dict
                prob = inner_dict[current_word]/total_value
                logged_prob = math.log(prob)
                summed_log_values += logged_prob
        else:
            return -math.inf
            
    return summed_log_values
             


def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    summed_log_values = 0.0
    
    for i in range(len(sequence)):
        current_word = sequence[i]
        inner_dicts = []  
        
        for model in models:
            #determine how much context this model needs
            
            for key in model.keys():
                current_length = len(key)  
                break
            
            context_length = current_length
            
            available_context = min(context_length, i)
            context = tuple(sequence[i - available_context:i])
            
            inner = query_n_gram(model, context)
            if inner is not None:
                inner_dicts.append(inner)
        
        if not inner_dicts:
            return -math.inf
        
        blended_prob_dict = blended_probabilities(inner_dicts)
        
        if current_word in blended_prob_dict:
            summed_log_values += math.log(blended_prob_dict[current_word])
        else:
            return -math.inf
    
    return summed_log_values

   
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
    
    print(log_likelihood_ramp_up(sequence[:20], models))
    

    # Task 4.2 test code
    
    print(log_likelihood_blended(sequence[:20], models))

