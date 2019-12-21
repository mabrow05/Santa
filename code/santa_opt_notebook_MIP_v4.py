

import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt

from mip.model import *


# In[2]:


# importing the data to take a look at what we have

df = pd.read_csv('../data/family_data.csv',index_col=0)

# Loading a feasible solution from another pass at solving the
# problem

#init_sol_file = 'submission_data/submission_exp_max_diff_30_w1_0_w2_0.4_w3_10.csv'
#init_sol = pd.read_csv(init_sol_file,index_col=0)



#init_sol_dict = {}

#for i,k in enumerate(init_sol):
#    days_list = [0 for i in range(100)]
#    days_list[k-1] = 1
#    init_sol_dict[i] = days_list

#df2.head()


# In[3]:


# load the penalties for days with high traffic
#day_penalty = pd.read_csv('day_penalty.txt',sep='\t',index_col=0)['penalty']
#day_penalty.head()


# In[4]:


#day_penalty[5]


# In[5]:


# for testing
num_days=20
num_families = 100
num_seconds = 300

df2 = df[:num_families].copy()

#people scaling
people_scale = df2.n_people.sum()/df.n_people.sum()
max_people = int(np.around(1.5*df2.n_people.sum()/num_days))
min_people = int(np.around(df2.n_people.sum()/2/num_days))

if num_days==100 and num_families==5000:
    max_people = 300
    min_people = 125

# day scaling
day_scale = num_days/100


# In[9]:


print(max_people)
print(min_people)
print(day_scale)
print(people_scale)


# In[10]:


if num_days!=100:
    for c in df2.columns.tolist()[:10]:
        df2[c] = np.random.randint(1,num_days+1,num_families)


init_sol = df2['choice_0']
# In[11]:


#df2.head()


# In[12]:


# I'm going to add a column which will represent the choice falling outside of any of the given choices.
# This will be choice_10, and it will be 101,,,


# In[13]:


print(df2.describe())


# In[14]:


#df2['choice_0'].hist(bins=[b for b in range(1,num_days+2,1)])


# In[20]:


lower_limit = min_people
upper_limit = max_people

# creating the decision variables
choice = ['choice_0','choice_1', 'choice_2', 'choice_3', 'choice_4', 'choice_5',
          'choice_6', 'choice_7', 'choice_8', 'choice_9','choice_10']
day = [i for i in range(1,num_days+1)]
fam_id = df2.index.tolist()
n_people = df2['n_people'].to_dict()
npd = [n for n in range(min_people,max_people+1,1)]


# In[16]:


print('Through loading data and setting up parameters.\n\nCreating Choice Matrix...')

choice_matrix = {}

for f in fam_id:

    if f%1000==0:
        print('{}/{}'.format(f,len(df2)))

    choice_matrix[f] = {}

    for d in day:

        choice_to_check = df2.loc[f,df2.columns.tolist()[:10]].values

        if d in choice_to_check:
            c_loc = np.where(choice_to_check==d)[0][0]
        else:
            c_loc = 10


        choice_matrix[f][d] = {c:(0 if i!=c_loc else 1) for i,c in enumerate(choice)}


# In[17]:


print('Creating Choice penalty mappers...')
# gift card contribution by choice
choice_gc = {}
choice_gc['choice_0'] = 0
choice_gc['choice_1'] = 50
choice_gc['choice_2'] = 50
choice_gc['choice_3'] = 100
choice_gc['choice_4'] = 200
choice_gc['choice_5'] = 200
choice_gc['choice_6'] = 300
choice_gc['choice_7'] = 300
choice_gc['choice_8'] = 400
choice_gc['choice_9'] = 500
choice_gc['choice_10'] = 500

# per member monetary contribution
choice_pm = {}
choice_pm['choice_0'] = 0
choice_pm['choice_1'] = 0
choice_pm['choice_2'] = 9
choice_pm['choice_3'] = 9
choice_pm['choice_4'] = 9
choice_pm['choice_5'] = 18
choice_pm['choice_6'] = 18
choice_pm['choice_7'] = 36
choice_pm['choice_8'] = 36
choice_pm['choice_9'] = 36+199
choice_pm['choice_10'] = 36+398


# Create a lookup table for the accounting penalty

# In[18]:


def accounting_penalty_actual(Nd,Nd1):
    diff = np.abs(Nd-Nd1)
    return 300/max_people*(Nd-min_people)/400 * Nd**(0.5+diff/50)


# In[19]:


print('Creating accounting penalty table...')
acc_table = {}
for Nd in npd:
    for Nd1 in npd:
        acc_table[(Nd,Nd1)] = accounting_penalty_actual(Nd,Nd1)


# In[ ]:


#acc_table[(300,125)]


# In[ ]:


#plt.plot(npd,[acc_table[(300,i)] for i in npd])


# The decision variable needs to be a boolean for each choice for each family. We will create a

# In[ ]:


# set the parameters

#max_diff = 35 # this is the maximum difference between two days in total number of people
#w1 = 0 # this is the weight applied to the simple difference ppd(d)-ppd(d+1)
#w2 = 0 # this is the weight applied to the simple linear shopping penalty
#w3 = 10 # this is the multiplier for the day penalty for high traffic days


# In[ ]:


print('Creating Model...\n')
# The prob variable is created to contain the problem data
m = Model()


# In[ ]:

print('Adding family decision variables...')
# The decision variables are actually the family and the day they are assigned
x = [ [m.add_var(name='fam_{},day_{}'.format(f,d),var_type=BINARY) for d in day] for f in fam_id ]


# In[ ]:

print('Adding number of people per day decision variables...')
y = [ [ [m.add_var(name='d_{}_nd_{}_nd1_{}'.format(d,nd,nd1),var_type=BINARY)#INTEGER, lb=min_people, ub=max_people)
         for nd1 in npd]
       for nd in npd]
     for d in day]


# In[ ]:


def ppd_fast(di):
    if di==num_days:
        di = num_days-1
    return xsum(x[fi][di]*n_people[f] for fi,f in enumerate(fam_id))

def ppd(d):
    if d>num_days:
        d=num_days
    di = day.index(d)
    return xsum(x[fi][di]*n_people[f] for fi,f in enumerate(fam_id))


# In[ ]:

print('Creating the objective function...')
m.objective = minimize(xsum(x[fi][di]*choice_matrix[f][d][c]*(choice_gc[c] + n_people[f]*choice_pm[c])
                       for c in choice for di,d in enumerate(day) for fi,f in enumerate(fam_id))
                      + xsum(y[di][ndi][nd1i]*acc_table[(nd,nd1)]
                             for nd1i,nd1 in enumerate(npd)
                             for ndi,nd in enumerate(npd)
                             for di,d in enumerate(day)))

#m.objective = minimize(xsum(x[fi][di]*choice_matrix[f][d][c]*(choice_gc[c] + n_people[f]*choice_pm[c])
#                            + ppd_fast(di)*(w1+w2) - w1*ppd_fast(di+1) - w2*125
#         for c in choice for di,d in enumerate(day) for fi,f in enumerate(fam_id)))

#m.objective = minimize(xsum(x[fi][di]*choice_matrix[f][d][c]*(choice_gc[c] + n_people[f]*choice_pm[c]+w3*day_penalty[d])
#                       for c in choice for di,d in enumerate(day) for fi,f in enumerate(fam_id)))

                      #+ xsum(ppd_fast(di)*(w1+w2) - w1*ppd_fast(di+1) - w2*125 for di,d in enumerate(day)))


# In[ ]:


# adding in the constraints

print('Creating constraint to ensure each family is assigned a single day...')
# The first set of constraints ensures each family only has a single day selected
for fi,f in enumerate(fam_id):
        m += xsum(x[fi][di] for di,d in enumerate(day)) == 1


# In[ ]:


# the second set of constraints guarantee that the total number of visitors is between 125 and 300 for
# for every single day leading up to christmas

print('Creating constraint to ensure the number of people per day does not exit bounds...')
for di,d in enumerate(day):
    m += ppd(d) >= min_people, ''
    m += ppd(d) <= max_people, ''


# In[ ]:

print('Creating constraints to make sure each day only has one entry for npd(d),npd(d+1) and\n \
        to make sure that the number of people matches between the two decision variables...')
for di,d in enumerate(day[:]):

    # each day should only have 1 entry
    m += xsum(y[di][ndi][nd1i] for nd1i,nd1 in enumerate(npd) for ndi,nd in enumerate(npd)) == 1

    # the number of people on day d needs to match
    m += ppd(d) == xsum(y[di][ndi][nd1i]*nd for nd1i,nd1 in enumerate(npd) for ndi,nd in enumerate(npd))


# In[ ]:


print('Creating constraint to make sure the number of people on d+1 matches from the \n \
        the first decision variable to the second decision variable...')
for di,d in enumerate(day[:-1]):
    # the number of people on the next day in the sum needs to match the next day
    m += ppd(d+1) == xsum(y[di][ndi][nd1i]*nd1 for nd1i,nd1 in enumerate(npd) for ndi,nd in enumerate(npd))

# the last day needs to have the next day set to the last day number of people
m += ppd(day[-1]) == xsum(y[-1][ndi][nd1i]*nd1 for nd1i,nd1 in enumerate(npd) for ndi,nd in enumerate(npd))


# In[29]:


# adding this third constraint to prevent the difference between each day from climbing too high.

#for di,d in enumerate(day[0:len(day)-1]):
#    m += ppd(d)-ppd(d+1) >= -max_diff, ''
#    m += ppd(d)-ppd(d+1) <= max_diff, ''


# In[ ]:
m.start = [(x[fi][init_sol[fi]-1], 1.0) for fi,f in enumerate(fam_id)]
m.threads = -1

#m.max_gap = 0.05
print('Solving.....')
status = m.optimize(max_seconds=num_seconds)

if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(m.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))


# In[ ]:


print('Solved with status {}'.format(status))


# In[ ]:


#m.write('model.lp')


# In[ ]:


obj = m.objective_value
print('Objective value: {}'.format(obj))


# In[ ]:


print('Creating family day dictionary from the resulting decision variable values...')
fam_day_dict = {}

for i,v in enumerate(m.vars[:int(len(x)*len(x[0]))]):

    if i%100000==0:
        print('{}/{}'.format(i,len(m.vars)))
    if abs(v.x) > 1e-6: # only printing non-zeros
        #print('{} : {}'.format(v.name, v.x))
        s = v.name.split(',')
        fam_day_dict[int(s[0][4:])] = int(s[1][4:])


# In[ ]:


sel_series = pd.Series(fam_day_dict,name='assigned_day')


# In[ ]:


#len(sel_series)


# In[ ]:


#sel_series.hist(bins=[b for b in range(1,num_days+2,1)])


# In[ ]:


#df2 = df2.join(sel_series)
df2['assigned_day'] = sel_series.astype(int)
#df2['assigned_day'] = df2.assigned_day.astype(int)
print(df2[['assigned_day']].head())


# In[ ]:


#df2[df2.assigned_day.isnull()]


# In[ ]:

print('Creating total people dict...')
total_people = {}
for d in day:
    mask = df2['assigned_day']==d
    total_people[d] = df2[mask].n_people.sum()
    print(total_people[d])


# Calculating the actual objective for the problem

# In[ ]:


def accounting_penalty_actual(Nd,diff):
    return 300/max_people*(Nd-min_people)/400 * Nd**(0.5+np.fabs(diff)/50)


# In[ ]:


total_accounting_penalty = sum([accounting_penalty_actual(total_people[d],total_people[d]-total_people[d+1])
                                if d<num_days
                                else accounting_penalty_actual(total_people[d],0)
                                for d in day])
print('Total accounting penalty: {}'.format(total_accounting_penalty))


# In[ ]:


# Adding a column to the dataframe for the choice made...

def choice_func(r):
    if r['assigned_day'] in r.values[:10]:
        return choice[list(r.values[:10]).index(r.assigned_day)]
    else:
        return 'choice_10'


# In[ ]:


df2['assigned_choice'] = df2.apply(choice_func,axis=1)
#df2[df2.assigned_choice=='choice_4'].head()


# In[ ]:


def simple_cost(r):
    return choice_gc[r['assigned_choice']] + r['n_people']*choice_pm[r['assigned_choice']]


# In[ ]:


total_simple_cost = df2.apply(simple_cost,axis=1).sum()
print('Simple cost: {}'.format(total_simple_cost))


# In[ ]:


final_score = total_simple_cost + total_accounting_penalty
print('Final Score: {}'.format(final_score))


# In[ ]:

df2.to_csv('optimal_data.txt',sep='|',header=True)
df2['assigned_day'].to_csv('submission_optimal.csv',header=True)
