from lxml import etree
import pandas as pd
import argparse

def create_pedflow(root, interval, direction, prob):
    child = etree.SubElement(root, "personFlow", id="%s_flow%s"%(direction,interval[0]), begin=str(interval[0]), end=str(interval[1]), \
                             probability=str(prob), departPos="-1")
    return child

def create_walk(child, fromEdge, toEdge):
    subchild = etree.SubElement(child, "walk", attr_from=fromEdge, to=toEdge)

parser = argparse.ArgumentParser()
parser.add_argument('--Date', default=None, type=str)
parser.add_argument('--Hour', default=None, type=int)
args = parser.parse_args()
Date = args.Date
Hour = args.Hour

pedcount = pd.read_csv('excel_files/pedcnt_weekday.csv', header=0)
test = pedcount[(pedcount['date']==Date)&(pedcount['hour']==Hour)].reset_index(drop=True)

if len(test)==0:
	raise ValueError('Date or Hour is out of data range.')
	
test['prob']=test['counts']/300
edges = ["gneE39", "687877404"]

root = etree.Element("routes")

for fivemin in range(12):
    interval = [300*fivemin, 300*(fivemin+1)]
    prob = float(test.loc[test['fivemin']==fivemin, 'prob'])
    for i in range(len(edges)):
        if i == 0:
            fromEdge = edges[0]
            toEdge = edges[1]
            direction = "nb"
        else:
            fromEdge = edges[1]
            toEdge = edges[0]
            direction = "sb"
        child0 = create_pedflow(root, interval, direction, prob/2) # equally set to nb and sb
        create_walk(child0, fromEdge, toEdge)

s = etree.tostring(root, pretty_print=True, encoding="unicode")
s2 = s.replace('attr_from','from')

with open(f'sumo_files/ped_files/ped.route.{Hour}.{Date}.xml', 'w') as f: 
    f.write(s2)