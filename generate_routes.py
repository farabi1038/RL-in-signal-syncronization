import subprocess
import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int)
args = parser.parse_args()
if args.run == None:
    print("enter run flag")
    exit()
np.random.seed(args.run)

run_num = args.run


x = pd.read_csv("excel_files/lanecnt_weekday.csv")
y = pd.read_csv("excel_files/pedcnt_weekday.csv")

x_date_idx = np.unique(x['date'])[:-1]
y_date_idx = np.unique(y['date'])

x_date_idx = np.delete(x_date_idx,np.where(x_date_idx == "2019-12-25")[0][0])

y_date_idx = np.delete(y_date_idx,np.where(y_date_idx == "2019-12-25")[0][0])

# for i in x_date_idx:
# 	x_hour = np.unique(x[(x['date']==i)]['hour'])
# 	y_hour = np.unique(y[(y['date']==i)]['hour'])
	
# 	try:
# 		np.where(x_hour == 7)[0][0]
# 	except IndexError:
# 		print(f"x {i} does not have 7am")

# 	try:
# 		np.where(x_hour == 12)[0][0]
# 	except IndexError:
# 		print(f"x {i} does not have 12pm")

# 	try:
# 		np.where(x_hour == 17)[0][0]
# 	except IndexError:
# 		print(f"x {i} does not have 5pm")

# 	try:
# 		np.where(y_hour == 7)[0][0]
# 	except IndexError:
# 		print(f"y {i} does not have 7am")

# 	try:
# 		np.where(y_hour == 12)[0][0]
# 	except IndexError:
# 		print(f"y {i} does not have 12pm")

# 	try:
# 		np.where(y_hour == 17)[0][0]
# 	except IndexError:
# 		print(f"y {i} does not have 5pm")

if not os.path.exists('temp_folder'):
    os.makedirs('temp_folder')
if not os.path.exists('sumo_files/route_files'):
    os.makedirs('sumo_files/route_files')
if not os.path.exists('sumo_files/ped_files'):
    os.makedirs('sumo_files/ped_files')

# args = ["python", "pedcnt"]
# subprocess.run(args)

for d in x_date_idx:
	for h in ["7", "12", "17"]:
		args = ["python", "utils/pedcnt_convert.py", "--Hour", h, "--Date", d]
		subprocess.run(args)
		for _ in range(1):

			seed1 = np.random.randint(1,999)
			seed2 = np.random.randint(1,999)

			args = ["python", 'C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py', "-n", "sumo_files/foothill.net.xml", "-s", str(seed1), "-p", "0.1", "-o", f"temp_folder/foothill.passenger.trips.{h}.{d}.{run_num}.seed1.{seed1}.seed2.{seed2}.xml", "-e", "3600", "--vehicle-class", "passenger", "--vclass", "passenger", "--prefix", "veh", "--min-distance", "100", "--trip-attributes", "departLane=\'best\'", "--fringe-start-attributes", "departSpeed=\'max\' departPos=\'last\'", "--allow-fringe.min-length", "100", "--lanes", "--validate", "--weights-prefix", "edge.edit"]
			subprocess.run(args)

			args = ["python", "utils/lanecnt_convert.py", "--Hour", h, "--Date", d, "--run", str(run_num), "--seed1", str(seed1), "--seed2", str(seed2)]
			subprocess.run(args)

			args = ["python", "C:/Program Files (x86)/Eclipse/Sumo/tools/routeSampler.py", "-t", f"temp_folder/edge.relation.{h}.{d}.{run_num}.seed1.{seed1}.seed2.{seed2}.xml", "-r", "routes.rou.xml", "-o", f"sumo_files/route_files/routes.sample.{h}.{d}.{run_num}.seed1.{seed1}.seed2.{seed2}.xml", "-s", str(seed2), "-a", "departPos=\'last\' departSpeed=\'max\'", "-f", "number"]
			subprocess.run(args)
