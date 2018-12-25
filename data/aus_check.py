import pickle

with open('aus.pkl', 'rb') as f:
    aus = pickle.load(f)

#print(aus)
#print(list(aus.keys()))
print(aus['N_0000000021_00562.jpg']/5)
