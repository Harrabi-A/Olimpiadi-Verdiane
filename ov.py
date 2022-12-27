from string import ascii_uppercase
import random
from random import randint


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import json


#check if two list have the same element at the same order
def listIsEqual(l1, l2):
	if (len(l1) == 0) and (len(l2)==0):
		return True
	if len(l1) != len(l2):
		return False
	else:
		for i in range(len(l1)):
			if l1[i] != l2[i]:
				return False
	return True

# Reuturn the index of element in list
def getPosInList(l,e):
	for i in range(len(l)):
		if l[i]==e:
			return i
	return -1

#Return the minimum element in the list
def getMinimum(l):
	if len(l)==0:
		return null
	min_val = l[0]
	for x in range(len(l)):
		if l[x]<min_val:
			min_val = l[x]
	return min_val

# Compute Objective Function 
def computeObjectiveFunction(alpa, teams, time,ns,nt_GS):
	tot = 0
	for s in range(ns):
		for t in range(nt_GS):
			if alpha.at[teams[s],time[t]] >=1:
				tot += alpha.at[teams[s],time[t]]
	return tot

def discipline_participation_t_s(X,Y,s,t):
	disc_part_st_list = []
	tmp_s_dg = X[s].copy()
	tmp_t_dg = Y[t].copy()
	for d in range(len(X[0])):
		for g in range(len(X[0][0])):
			if (tmp_s_dg[d][g]*tmp_t_dg[d][g]) != 0:
				disc_part_st_list.append(d)
	return disc_part_st_list

# Check if the discipline team belong to restricted list, if so return False, otherwise True
def checkRestriction(t_option, t_change_option, inbound_restriction, outbound_restriction):
	for i in range(len(t_option)):
		if t_option[i] in outbound_restriction:
			return False
	for j in range(len(t_change_option)):
		if t_change_option[j] in inbound_restriction:
		    return False
	return True

#Return the index of the discipline with maximum value in slot t in matrix m
def max_val_pos(m,t, option_list):
	max_val = 0
	max_pos = -1
	tmp_list = m.loc[:,time[t]]
	for i in range(len(option_list)):
		print(option_list[i], " " ,tmp_list.loc[option_list[i]].sum())
		if tmp_list.loc[option_list[i]].sum() > max_val:
			max_val = tmp_list.loc[option_list[i]].sum()
			max_pos = i
	return max_pos

# Return an index of the discipline which contains the team group passed as an argument
def find_d_index(discipline, group_teams):
	for d in range(len(discipline)):
		for g in range(len(discipline[d])):
			if(listIsEqual(group_teams, discipline[d][g])):
			#if(group_teams == discipline[d][g]).all():
				return d
	return -1

# Return the index (order) of team group in the discipline passed as an argument
def find_d_group_index(discipline, d_index, group_teams):
	for g in range(len(discipline[d_index])):
		if(listIsEqual(group_teams, discipline[d_index][g])):
		#if (group_teams == discipline[d_index][g]).all():
			return g
	return -1

ns = int(input("How many teams?\n"))
teams = []
r = 0
while len(teams) != ns :
	t = ascii_uppercase[r]
	for x in ascii_uppercase:
		if len(teams) == ns:
			break
		else:
		    team = t + x
		    #print(team)
		    teams.append(team)

	r = r+1

print(teams)


Np = int(input("how many teams per group?\n"))
if ns % Np != 0:
	print("Teams can't be divided into equal groups !!!")
	exit()

Ng = int(ns/Np)


print("Group stages requires no less than ", Ng, "slots (minimum required to guarantee constraint-1)")
nt_GS = int(input("how many slot to allocate in group stage\n"))
if nt_GS < Ng :
    print("No Solution: insufficient number of slots in group stage")
    exit()

w = int(input("What is the maximum number of overlaps?\n"))

nd = int(input("how many disciplines?\n"))

discipline_name = []
for i in range(nd):
	tmp_discipline_name="D"+str(i)
	discipline_name.append(tmp_discipline_name)

discipline = []
timeline = []
for i in range(nd):
	tmp_teams = teams.copy()
	random.shuffle(tmp_teams)
	print(tmp_teams)
	groups_discipline = []
	timeline_discipline = []
	while len(tmp_teams) != 0 :
		groups_discipline.append(tmp_teams[:Np])
		timeline_discipline.append(tmp_teams[:Np])
		tmp_teams = tmp_teams[Np:]

	discipline.append(groups_discipline)

	while len(timeline_discipline) != nt_GS:
		timeline_discipline.append([])

	timeline.append(timeline_discipline)


print(discipline)
print(timeline)
timeline = np.array(timeline, dtype='object')
print("Groups by Discipline:")
print(discipline)
print("Timeline for each discipline")
print(timeline)

# Initialize the matrix X
X = []
for i in range(ns):
	X_team = []
	for j in range(nd):
		X_team_discipline = []
		while len(X_team_discipline) != Ng:
			X_team_discipline.append(0)
		X_team.append(X_team_discipline)

	X.append(X_team)

print(X)

# Initialize the matrix Y
Y = []
for i in range(nt_GS):
	Y_slot = []
	for j in range(nd):
		Y_slot_discipline = []
		while len(Y_slot_discipline) != Ng:
			Y_slot_discipline.append(0)
		Y_slot.append(Y_slot_discipline)

	Y.append(Y_slot)

print(Y)

# Initialize empty alpha matrix
time = []
for x in range(nt_GS):
	tmp_char = "T"+ str(x)
	time.append(tmp_char)

alpha = pd.DataFrame(columns=time, index=teams)
Z = pd.DataFrame(columns=time, index=teams)


# Fill X matrix
for s in range(ns):
	for d in range(nd):
		for g in range(Ng):
			if (teams[s] in discipline[d][g]):
				X[s][d][g] = 1
			else:
				X[s][d][g] = 0

print(X)

# Fill Y matrix
for t in range(nt_GS):
	for d in range(nd):
		for g in range(Ng):
			if (listIsEqual(discipline[d][g],timeline[d][t])):
				Y[t][d][g] = 1
			else:
				Y[t][d][g] = 0

print(Y)

# Compute alpha matrix
for s in range(ns):
	for t in range(nt_GS):
		tmp_s_dg = X[s].copy()
		tmp_t_dg = Y[t].copy()
		alpha_s_t = 0
		for d in range(nd):
			for g in range(Ng):
				alpha_s_t = alpha_s_t + tmp_s_dg[d][g]*tmp_t_dg[d][g]
		alpha.at[teams[s],time[t]] = alpha_s_t-1
		if alpha_s_t-1 >= 1:                                                       #   Problem Variable (1= overlap )
			Z.at[teams[s],time[t]] = alpha_s_t-1
		else:
			Z.at[teams[s],time[t]] = 0

print(alpha)
#print(Z)
z= Z.sum().sum()
print("Z = ", z)

#
max_overlaps = alpha.max().max()
if max_overlaps <= w:
	admissible = True
	print("Admissible solution")

alpha_t = alpha.sum()
print(alpha_t)
sum_alpha= alpha_t.sum()                                     # Sum_alpha = minimum bound under which the problem has no solution, if w*ns < Sum_alpha
if w*ns < sum_alpha:
	print("Problem has no solution")
	#exit()


i = 0
while True:
	#print("****** iteration #",i," *********\n")
	max_overlaps = alpha.max().max()
	print("FO = max overlaps = ",max_overlaps)
	if max_overlaps <= w:
		print("FEASIBLE SOLUTION")
		break
	else:
		#find the max and exchange two team in the same discipline in different group following the defined policie
		for s in range(ns):
			for t in range(nt_GS):
				if (alpha.at[teams[s],time[t]])==max_overlaps:
					s_change1 = teams[s]
					t_change1 = time[t]
					print("Selected team with the maximum overlaps: ",s_change1," at T",t)
					#find t_change2 option (1- where s_change1 have the lowest overlaps)
					Ti = alpha.loc[:, time[t]]
					s_change2_options = Ti.where(Ti == Ti.min()).dropna(how='all').index.values.tolist()
					print("option teams ",s_change2_options)                   #TODO check case min=max
					Si = alpha.loc[teams[s],:]
					t_change2_options = Si.where(Si == Si.min()).dropna(how='all').index.values.tolist()
					print("option slots ",t_change2_options)
					alpha_options = pd.DataFrame(columns=t_change2_options, index=s_change2_options)
					#check the existence of solution
					if len(s_change2_options) != 0:
						i += 1
						print("****** Feasibility change #",i," *********\n")
					else:
						break
					#fill alpha_options
					for s_tmp in range(len(s_change2_options)):
						for t_tmp in range(len(t_change2_options)):
							alpha_options.at[s_change2_options[s_tmp],t_change2_options[t_tmp]] = alpha.at[s_change2_options[s_tmp],t_change2_options[t_tmp]]

					
					print(alpha_options)
					alpha_options = alpha_options.where(alpha_options == alpha_options.max().max()).dropna(how='all')
					alpha_options = alpha_options.where(alpha_options == alpha_options.max().max()).dropna(axis=1 ,how='all')
					t_change2 = alpha_options.columns.values.tolist()[0]
					alpha_options= alpha_options.iloc[:, 0]
					s_change2 = alpha_options.dropna(how='all').index.values.tolist()[0]
					print("team to exchange ",s_change2)
					print("slot to exchange ",t_change2)
					#find discipline
					d_change = -1
					for d in range(len(discipline)):
						if (s_change1 in discipline[d][getPosInList(time,t_change1)]) and (s_change2 in discipline[d][getPosInList(time,t_change2)]):
							d_change = d
							print("discipline D",d_change)
							break
					print(t_change1,"  ",s_change1,"  <------->  ",s_change2,"  ",t_change2)
					#update discipline matrix and timline
					#TODO discipline may have out of range errors
					discipline[d][getPosInList(time,t_change2)][getPosInList(discipline[d][getPosInList(time,t_change2)],s_change2)]=s_change1
					discipline[d][getPosInList(time,t_change1)][getPosInList(discipline[d][getPosInList(time,t_change1)],s_change1)]=s_change2
					#print(discipline)

					s1_pos = getPosInList(timeline[d][getPosInList(time,t_change1)],s_change1)
					#print("position of s1 ",s1_pos)
					s2_pos = getPosInList(timeline[d][getPosInList(time,t_change2)],s_change2)
					#print("position of s2 ",s2_pos)

					tmp_list = timeline[d][getPosInList(time,t_change2)]
					tmp_list[s2_pos] = s_change1
					#print("group in tchange2 ",tmp_list)
					timeline[d][getPosInList(time,t_change2)]= tmp_list


					tmp_list2 = timeline[d][getPosInList(time,t_change1)]
					tmp_list2[s1_pos] = s_change2
					#print("group in tchange1 ",tmp_list2)
					timeline[d][getPosInList(time,t_change1)]= tmp_list2
					print(timeline)

					#update X matrix
					X[s][d][t] = 0
					X[s][d][getPosInList(time,t_change2)] = 1
					X[getPosInList(teams,s_change2)][d][t] = 1
					X[getPosInList(teams,s_change2)][d][getPosInList(time,t_change2)] = 0
					
					#update alpha
					#TODO update Z (not required in this part)
					alpha.at[s_change1,t_change1] = alpha.at[s_change1,t_change1]-1
					alpha.at[s_change1,t_change2] = alpha.at[s_change1,t_change2]+1
					alpha.at[s_change2,t_change1] = alpha.at[s_change2,t_change1]+1
					alpha.at[s_change2,t_change2] = alpha.at[s_change2,t_change2]-1
					print(alpha)
					print("Objective Function [OLD",z,"]")
					z = computeObjectiveFunction(alpha, teams, time, ns, nt_GS)
					print("Objective Function [NEW",z,"]")


					#input("press enter to continue")


#export result
'''
f = open("schedule.txt", "a")
discipline_txt = "******* Discipline D{} ********\n"
group_txt = "G{}: {} - {} - {} - {}\n"
for d in range(nd):
	f.write(discipline_txt.format(d))
	tmp_txt = ''
	for g in range(Ng):
		tmp_txt = tmp_txt + group_txt.format(g,discipline[d][g][0],discipline[d][g][1],discipline[d][g][2],discipline[d][g][3])

	f.write(tmp_txt)
		

f.close()'''

# Optimazation algorithm
print("\n\n\nOptimazation Phase...")

# matrices of distance
regions = ['R1','R2','R3','R4','R5']
distance = [[0,1,4,5,5],
            [1,0,2,3,5],
            [4,2,0,5,10],
            [3,3,5,0,2],
            [5,5,10,2,0]]
max_distance = 10


#allocate for each discipline their regions
disciplineRegions = []
for d in range(nd):
	# for debug pupose discipline regions will be allocated randomly
	disciplineRegions.append(regions[randint(0, 4)])
print("Discpline Regions: ", disciplineRegions)

# Initialize void beta matrix
beta = pd.DataFrame(columns=time, index=teams)
for s in range(ns):
	for t in range(nt_GS):
		if t == 0:
			#ASSUMPTION first time slot has no commute problem
			beta.at[teams[s],time[t]]=0;
		else:
			# Compute number of overlaps
			disc_part_st = discipline_participation_t_s(X,Y,s,t)
			#print("***DEBUG***      squad ",teams[s]," at ",time[t]," participate in discipline ",disc_part_st)
			disc_part_prev_st = discipline_participation_t_s(X,Y,s,t-1)
			#print("***DEBUG***      squad ",teams[s]," at ",time[t-1]," particape in discipline ",disc_part_prev_st,"")

			if (len(disc_part_prev_st) == 0) or (len(disc_part_st ) == 0):
				beta.at[teams[s],time[t]] = 0
				continue

			selected_disc_st = -1
			selected_disc_prev_st = -1
			#Select max distance for each discipline in t_prev with discipline in t
			tmp_distance_list = []
			tmp_discipline_list = []
			tmp_max_distance = 0
			tmp_pos_max_distance = -1
			for x in range(len(disc_part_st)):
				region_x = disciplineRegions[disc_part_st[x]]
				for y in range(len(disc_part_prev_st)):
					region_y = disciplineRegions[disc_part_prev_st[y]]
					if distance[x][y]>tmp_max_distance:
						tmp_max_distance = distance[x][y]
						tmp_pos_max_distance = y

				tmp_distance_list.append(tmp_max_distance)
				tmp_discipline_list.append(tmp_pos_max_distance)

			tmp_min_distance = getMinimum(tmp_distance_list)
			tmp_pos_min_distance = getPosInList(tmp_distance_list, tmp_min_distance)
			selected_disc_st = disc_part_st[tmp_pos_min_distance]
			selected_disc_prev_st = tmp_distance_list[tmp_pos_min_distance]
			beta.at[teams[s],time[t]] = distance[getPosInList(regions,disciplineRegions[selected_disc_st])][getPosInList(regions,disciplineRegions[selected_disc_prev_st])]			

i=0
while True:
	#print("****** iteration #",i," *********\n")
	print(alpha)
	print(beta)
	OF_alpha = alpha.sum().sum()
	OF_beta = beta.sum().sum()
	print("Objective Function alpha = ",OF_alpha)
	print("Objective Function beta = ", OF_beta)
	z = computeObjectiveFunction(alpha, teams, time, ns, nt_GS)
	print("Z= OF_alpha + 0*(OF_beta) =",z)
	if (z == (nd-nt_GS)*ns):
		print("OPTIMUM SOLUTION")
		break
	else:
		#find the max and exchange two team in the same discipline in different group following the defined policie
		for s in range(ns):
			for t in range(nt_GS):
				if (alpha.at[teams[s],time[t]])==-1:
					s_change1 = teams[s]
					t_change1 = time[t]
					print("Selected team with -1 overlaps: ",s_change1," at T",t)
					Ti = alpha.loc[:, time[t]]
					print(Ti)
					s_change2_options = Ti.where(Ti == Ti.max()).dropna(how='all').index.values.tolist()   # the max must be different than -1
					print("option teams ",s_change2_options)
					Si = alpha.loc[teams[s],:]
					t_change2_options = Si.where(Si == Si.max()).dropna(how='all').index.values.tolist()
					print("option slots ",t_change2_options)
					alpha_options = pd.DataFrame(columns=t_change2_options, index=s_change2_options)
					beta_options = pd.DataFrame(columns=t_change2_options, index=s_change2_options)
					if len(s_change2_options) != 0:
						i += 1
						print("****** Optimazation change #",i," *********\n")
					else:
						break
					#fill alpha_options
					for s_tmp in range(len(s_change2_options)):
						for t_tmp in range(len(t_change2_options)):
							alpha_options.at[s_change2_options[s_tmp],t_change2_options[t_tmp]] = alpha.at[s_change2_options[s_tmp],t_change2_options[t_tmp]]
							beta_options.at[s_change2_options[s_tmp],t_change2_options[t_tmp]] = beta.at[s_change2_options[s_tmp],t_change2_options[t_tmp]]
					#print(alpha_options)
					#print(beta_options)
					'''two possible flow:
					1- choose the best to reduce alpha
					2- choose the best to reduce beta
					In the code below, we proceed to reduce alpha and obtain the optimum solution'''
					alpha_options = alpha_options.where(alpha_options == alpha_options.min().min()).dropna(how='all')
					alpha_options = alpha_options.where(alpha_options == alpha_options.min().min()).dropna(axis=1 ,how='all')
					t_change2 = alpha_options.columns.values.tolist()[0]
					alpha_options= alpha_options.iloc[:, 0]
					s_change2 = alpha_options.dropna(how='all').index.values.tolist()[0]
					print("team to exchange ",s_change2)
					print("slot to exchange ",t_change2)
					#find discipline
					d_change = -1
					for d in range(len(discipline)):
						if (s_change2 in discipline[d][getPosInList(time,t_change1)]) and (s_change1 in discipline[d][getPosInList(time,t_change2)]):
							d_change = d
							print("discipline D",d_change)
							break
					print(t_change1,"  ",s_change2,"  <------->  ",s_change1,"  ",t_change2)
					discipline[d][getPosInList(time,t_change2)][getPosInList(discipline[d][getPosInList(time,t_change2)],s_change2)]=s_change2
					discipline[d][getPosInList(time,t_change1)][getPosInList(discipline[d][getPosInList(time,t_change1)],s_change1)]=s_change1
					#print(discipline)

					s2_pos = getPosInList(timeline[d][getPosInList(time,t_change1)],s_change2)
					#print("position of s1 ",s1_pos)
					s1_pos = getPosInList(timeline[d][getPosInList(time,t_change2)],s_change1)
					#print("position of s2 ",s2_pos)

					tmp_list = timeline[d][getPosInList(time,t_change2)]
					tmp_list[s1_pos] = s_change2
					#print("group in tchange2 ",tmp_list)
					timeline[d][getPosInList(time,t_change2)]= tmp_list


					tmp_list2 = timeline[d][getPosInList(time,t_change1)]
					tmp_list2[s2_pos] = s_change1
					#print("group in tchange1 ",tmp_list2)
					timeline[d][getPosInList(time,t_change1)]= tmp_list2
					print(timeline)

					#update X matrix
					X[s][d][t] = 0
					X[s][d][getPosInList(time,t_change2)] = 1
					X[getPosInList(teams,s_change2)][d][t] = 1
					X[getPosInList(teams,s_change2)][d][getPosInList(time,t_change2)] = 0
					
					#update alpha
					alpha.at[s_change2,t_change1] = alpha.at[s_change2,t_change1]-1
					alpha.at[s_change2,t_change2] = alpha.at[s_change2,t_change2]+1
					alpha.at[s_change1,t_change1] = alpha.at[s_change1,t_change1]+1
					alpha.at[s_change1,t_change2] = alpha.at[s_change1,t_change2]-1
					print("Objective Function [OLD",z,"]")
					z = computeObjectiveFunction(alpha, teams, time, ns, nt_GS)
					print("Objective Function [NEW",z,"]")
			
	input("press enter to continue")

'''
# Initialize void beta matrix
beta = pd.DataFrame(columns=time, index=teams)
for s in range(ns):
	for t in range(nt_GS):
		if t == 0:
			#ASSUMPTION first time slot has no commute problem
			beta.at[teams[s],time[t]]=0;
		else:
			# Compute number of overlaps
			disc_part_st = discipline_participation_t_s(X,Y,s,t)
			#print("***DEBUG***      squad ",teams[s]," at ",time[t]," participate in discipline ",disc_part_st)
			disc_part_prev_st = discipline_participation_t_s(X,Y,s,t-1)
			#print("***DEBUG***      squad ",teams[s]," at ",time[t-1]," particape in discipline ",disc_part_prev_st,"")

			if (len(disc_part_prev_st) == 0) or (len(disc_part_st ) == 0):
				beta.at[teams[s],time[t]] = 0
				continue

			selected_disc_st = -1
			selected_disc_prev_st = -1
			#Select max distance for each discipline in t_prev with discipline in t
			tmp_distance_list = []
			tmp_discipline_list = []
			tmp_max_distance = 0
			tmp_pos_max_distance = -1
			for x in range(len(disc_part_st)):
				region_x = disciplineRegions[disc_part_st[x]]
				for y in range(len(disc_part_prev_st)):
					region_y = disciplineRegions[disc_part_prev_st[y]]
					if distance[x][y]>tmp_max_distance:
						tmp_max_distance = distance[x][y]
						tmp_pos_max_distance = y

				tmp_distance_list.append(tmp_max_distance)
				tmp_discipline_list.append(tmp_pos_max_distance)

			tmp_min_distance = getMinimum(tmp_distance_list)
			tmp_pos_min_distance = getPosInList(tmp_distance_list, tmp_min_distance)
			selected_disc_st = disc_part_st[tmp_pos_min_distance]
			selected_disc_prev_st = tmp_distance_list[tmp_pos_min_distance]
			beta.at[teams[s],time[t]] = distance[getPosInList(regions,disciplineRegions[selected_disc_st])][getPosInList(regions,disciplineRegions[selected_disc_prev_st])]			

print("NEW beta ", beta.sum().sum())'''

'''
# Open the CSV file for reading
with open('DisciplineName.csv', 'r') as file:
  # Create a CSV reader object
  reader = csv.reader(file)
  
  # Read the first row from the file (the data row)
  data_row = next(reader)
  
  # Convert the data row to a list
  NomiDiscipline = list(data_row)

print(NomiDiscipline)

if (len(NomiDiscipline) != nd):
	print("number of discipline in NomiDiscipline is different than the input")
	print(len(NomiDiscipline))
	input("INCONSISTENCY ERROR")


# Export to JSON
disciplien_toJSON_list = []
discipline_dict = {
	
}

timeline_toJSON_list = []
for x in range(nt_GS):
	timeline_dict = {
		"slot":time[x],
	}
	for y in range(nd):
		timeline_dict[NomiDiscipline[y]] = discipline[y][x]
		#timeline_dict[discipline_name[y]]=discipline[y][x]
	timeline_toJSON_list.append(timeline_dict)

with open("timeline_2.json", "w") as outfile:
	json.dump(timeline_toJSON_list, outfile)'''




# compute the coupling between teams
q_sum = 0
q_min = 9999
q_max = 0
q_matrix = []
for i in range(ns):
	row = []
	for j in range(ns):
		p = 0
		if i !=j :
			for t in range(nt_GS):
				for d in range(nd):					
					if (teams[i] in discipline[d][t]) and (teams[j] in discipline[d][t]):
						p += 1
			#eventually update min and max
			if p > q_max:
				q_max = p
			if p < q_min:
				q_min = p
			q_sum += p
		row.append(p)
	q_matrix.append(row)

q = np.matrix(q_matrix)
print(q)

q_avg = q_sum / (ns*ns-ns)
print("The average coupling between teams is q=",q_avg)
plt.matshow(q)
plt.show()

#compute q_mae
q_tot_error = 0
for i in range(ns):
	for j in range(ns):
		if i == j:
			continue
		q_tot_error = q_tot_error + abs(q.item((i,j)) - q_avg)

q_mae = q_tot_error / (ns*ns-ns)
print("low value of MAE corrsipond to a better distribution equality in teams coupling")
print("mean absolute error: ",q_mae)
print("\n\n***Solution***")
print("OF: ",OF_alpha)
print("alpha: ",OF_alpha)
print("beta:", "NOT IMPLEMENTED YET")
print("\n***quality of solution***")
print("coupling AVG: ",q_avg)
print("coupling MAE: ",q_mae)
print("coumpling MAX: ",q_max)
print("coupling MIN: ",q_min)
