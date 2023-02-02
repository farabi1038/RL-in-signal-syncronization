from lxml import etree
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Date', default=None, type=str)
parser.add_argument('--Hour', default=None, type=int)
parser.add_argument('--run', default=None, type=int)
parser.add_argument('--seed1', default=None, type=int)
parser.add_argument('--seed2', default=None, type=int)
args = parser.parse_args()
Date = args.Date
Hour = args.Hour
run_num = args.run
seed1 = args.seed1
seed2 = args.seed2

def create_interval(root, interval):
    child = etree.SubElement(root, "interval", begin=str(interval[0]), end=str(interval[1]))
    return child

def create_edgeRelation(child, fromEdge, toEdge, cnts):
    subchild = etree.SubElement(child, "edgeRelation", attr_from=fromEdge, to=toEdge, count=str(cnts))

def interp(temp, order2):
    temp = pd.merge(temp, order2, on=['SignalID','Direction'], how='outer')
    temp[['MovementType','Phase','date','hour','fivemin']] = temp[['MovementType','Phase','date','hour','fivemin']].fillna(method='ffill')
    temp= temp.sort_values(by='order').reset_index(drop=True)
    temp['counts']=temp['counts'].interpolate(method='linear')
    temp0=temp[temp['edited']==0]
    temp0=temp0.drop('edited', axis=1)
    temp1=temp[temp['edited']==1]
    left=temp1.drop('edited', axis=1)
    left['MovementType']='Left'
    left['counts']=left['counts']*0.1
    thru=temp1.drop('edited', axis=1)
    new_temp=pd.concat([temp0,left,thru])
    return new_temp

def data_processing(raw_cnt, uturn, edges, Date, Hour):
	thru = raw_cnt[(raw_cnt['MovementType']=='Thru-Left')].reset_index(drop=True)
	left = raw_cnt[(raw_cnt['MovementType']=='Thru-Left')].reset_index(drop=True)
	thru['MovementType']='Thru'
	thru['counts']*=0.9
	left['MovementType']='Left'
	left['counts']*=0.1
	temp = pd.concat([thru, left])
	raw_cnt_0 = raw_cnt[(raw_cnt['MovementType']!='Thru-Left')].reset_index(drop=True)
	lanecnt = pd.concat([raw_cnt_0, temp])

	test = lanecnt[(lanecnt['date']==Date)&(lanecnt['hour']==Hour)].reset_index(drop=True)
	test2 = test[test['Phase']==2]
	test6 = test[test['Phase']==6]
	rest = test[(test['Phase']!=2)&(test['Phase']!=6)]

	order2 = pd.DataFrame()
	order2['SignalID']=[7371,7223,7222,7221,7220,7503,7219,7218,7217,7216]
	order2['order']=range(10)
	order2['Direction']=['NB','NB','NB','WB','NB','NB','NB','NB','NB','WB']
	order2['edited']=[0,1,1,1,0,1,0,0,0,0]
	new_test2=[]
	for i in range(12):
		temp = test2[test2['fivemin']==i]
		temp = interp(temp, order2)
		new_test2.append(temp)
	new_test2=pd.concat(new_test2)

	order6 = pd.DataFrame()
	order6['SignalID']=[7216,7217,7218,7219,7503,7220,7221,7222,7223,7371]
	order6['order']=range(10)
	order6['Direction']=['EB','SB','SB','SB','SB','SB','EB','SB','SB','SB']
	order6['edited']=[0,0,0,0,1,0,1,1,1,0]
	new_test6=[]
	for i in range(12):
		temp = test6[test6['fivemin']==i]
		temp = interp(temp, order6)
		new_test6.append(temp)
	new_test6=pd.concat(new_test6)

	res=pd.concat([new_test2,new_test6])
	res = res.drop(['order'], axis=1)
	allres = pd.concat([res,rest])
	allres=pd.merge(allres,uturn,on=['SignalID','Direction','MovementType'], how='left')
	allres=allres.fillna(0)
	left = allres[allres['Uturn']==1]
	uturn = allres[allres['Uturn']==1]
	others = allres[allres['Uturn']==0]
	left['counts']*=0.9
	uturn['counts']*=0.1
	uturn['MovementType']='Uturn'
	final = pd.concat([others,left,uturn])
	final = final.drop('Uturn', axis=1)

	final = pd.merge(edges, final, on=['SignalID','Direction','MovementType'], how='left')
	final = final.dropna()
	final['counts']=final['counts'].round(0).astype('int')
	return final

def data_correction(test):
	test['remove'] = 0
	test.loc[((test['SignalID']==7503)| (test['SignalID']==7222)|(test['SignalID']==7223)) & (test['MovementType']=='Thru'),'remove']=1
	test.loc[(test['SignalID']==7221) & ((test['Phase']==2)|(test['Phase']==6)),'remove']=1
	test = test[test['remove']==0]
	return test

# inputs
raw_cnt = pd.read_csv('excel_files/lanecnt_weekday.csv', header=0)
uturn = pd.read_csv('excel_files/lanecnt_uturn.csv', header=0)
edges = pd.read_csv('excel_files/lanecnt_sumo_edges.csv', header=0)

# count process
data = data_processing(raw_cnt, uturn, edges, Date, Hour)
data = data_correction(data)

#convert to xml
root = etree.Element("edgeRelations")
for fivemin in range(12):
#     interval = [300*fivemin, 300*(fivemin+1)] # only 5pm used this
    interval = [280*fivemin, 280*(fivemin+1)]
    child0 = create_interval(root, interval)
    temp0 = data[data['fivemin']==fivemin].reset_index(drop=True)
    
    for i in range(len(temp0)):
        fromEdge = temp0.loc[i]['FromEdge'] 
        toEdge = temp0.loc[i]['ToEdge']
        counts = int(temp0.loc[i, 'counts'])
        create_edgeRelation(child0, fromEdge, toEdge, counts)

s = etree.tostring(root, pretty_print=True, encoding="unicode")
s2 = s.replace('attr_from','from')
with open(f'temp_folder/edge.relation.{Hour}.{Date}.{run_num}.seed1.{seed1}.seed2.{seed2}.xml', 'w') as f: 
    f.write(s)