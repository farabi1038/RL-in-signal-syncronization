import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import os

def get_waiting_time(df, meta, WEIGHT):
    df=df[['edge_id','edge_left','edge_waitingTime']]
    res = pd.merge(df, meta, left_on='edge_id', right_on='Edge')
    res = res.drop_duplicates()
    res['edge_waitingTime']=res['edge_waitingTime'].astype('double')
    res['edge_left']=res['edge_left'].astype('int')
    res['avg_waiting']=res['edge_waitingTime']/res['edge_left']
    res = res.groupby(['Main'])['avg_waiting'].mean().reset_index()
    final = float(res.loc[res['Main']==1, 'avg_waiting'])*(1-WEIGHT)+float(res.loc[res['Main']==0, 'avg_waiting'])*WEIGHT
    return final

def check_start(target_edges, trips):
    checked = 0
    for e in target_edges:
        if e in trips:
            checked = 1
            break
    return checked

def check_through(edges, phaseid, trips):
    checked = 0
    temp = edges[edges['Phase']==phaseid]
    if phaseid == 2:
        temp = temp[(temp['SignalID']!=7371)].reset_index(drop=True)
    else:
        temp = temp[(temp['SignalID']!=7216)].reset_index(drop=True)
    count = 0
    for i in range(len(temp)):
        e = temp.loc[i, 'Edge']
        if e in trips:
            count += 1
    if count == len(temp):
        checked = 1
    return checked

def get_thru_id(trips, edges):
    trips = trips.iloc[:,3:]
    trips = trips.dropna()
    start_edges_6 = list(edges.loc[(edges['SignalID']==7216)&(edges['Phase']==6), 'Edge'])
    start_edges_2 = list(edges.loc[(edges['SignalID']==7371)&(edges['Phase']==2), 'Edge'])
    trips['start_2'] = trips.apply(lambda x: check_start(start_edges_2, x.route_edges), axis=1)
    trips['start_6'] = trips.apply(lambda x: check_start(start_edges_6, x.route_edges), axis=1)
    trips['through_2'] = trips.apply(lambda x: check_through(edges, 2, x.route_edges), axis=1)
    trips['through_6'] = trips.apply(lambda x: check_through(edges, 6, x.route_edges), axis=1)
    thru_trips_2 = trips[((trips['start_2']==1)&(trips['through_2']==1))]
    thru_trips_6 = trips[((trips['start_6']==1)&(trips['through_6']==1))]

    thru_trips_id_2 = list(thru_trips_2['flow_id'])
    thru_trips_id_6 = list(thru_trips_6['flow_id'])
    
    return thru_trips_id_2, thru_trips_id_6

def thru_trip_match(thru_trips_id, trip_id):
    checked = 0
    for t in thru_trips_id:
        if t in trip_id:
            checked = 1
            break
    return checked

def get_num_stops(tripinfo, thru_trips_id_2, thru_trips_id_6):
    tripinfo = tripinfo.loc[:, tripinfo.columns.str.startswith('tripinfo')] # some may not have walk info, use index will fail
    tripinfo = tripinfo.dropna(subset=['tripinfo_arrival'])
    tripinfo['thru_trip_2']=tripinfo.apply(lambda x: thru_trip_match(thru_trips_id_2, x.tripinfo_id), axis=1)
    tripinfo['thru_trip_6']=tripinfo.apply(lambda x: thru_trip_match(thru_trips_id_6, x.tripinfo_id), axis=1)
    tripinfo_2 = tripinfo[tripinfo['thru_trip_2']==1]
    tripinfo_6 = tripinfo[tripinfo['thru_trip_6']==1]
    if len(tripinfo_2)>0:
        num_stop_2 = np.mean(tripinfo_2.tripinfo_waitingCount)
    else:
        num_stop_2 = 0
    if len(tripinfo_6)>0:
        num_stop_6 = np.mean(tripinfo_6.tripinfo_waitingCount)
    else:
        num_stop_6 = 0
    return num_stop_2, num_stop_6

def evaluate(TOD, WEIGHT, edges, thru_trips_id_2, thru_trips_id_6):

    os.system('out_%s.bat'%TOD)
    metrics = pd.read_csv('metric.res.%s.csv'%TOD, header=0)
    wt = get_waiting_time(metrics, edges, WEIGHT)

    tripinfo = pd.read_csv('tripinfo.%s.csv'%TOD, header=0)
    num_stop_2, num_stop_6 = get_num_stops(tripinfo, thru_trips_id_2, thru_trips_id_6)

    return -(wt+num_stop_2+num_stop_6)