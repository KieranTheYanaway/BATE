import argparse
import os
import math
from matplotlib import pyplot as plt
import numpy as np
from obspy import UTCDateTime, read, read_inventory, signal
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from datetime import timedelta
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, CrosshairTool, RangeTool, ColumnDataSource, DatetimeTickFormatter, HoverTool
from bokeh.plotting import figure, curdoc
from bokeh.events import Tap, DoubleTap
from bokeh.models import ColumnDataSource
from bokeh.models import LinearColorMapper
from bokeh.palettes import Inferno
from pathos.multiprocessing import ProcessingPool as Pool
from concurrent.futures import ThreadPoolExecutor
from obspy.signal.trigger import classic_sta_lta
from scipy.ndimage.filters import gaussian_filter1d, maximum_filter1d

def get_data(lat, lon, starttime, length, maxdistance, provider, name):
    """
    Download seismic data using the specified parameters.
    """
    cd = CircularDomain(latitude=lat, longitude=lon, minradius=0.0, maxradius=maxdistance / 111.12)
    res = Restrictions(
        starttime=UTCDateTime(starttime) - length,
        endtime=UTCDateTime(starttime) + 2 * length,
        network="*",
        station="*",
        location="*",
        reject_channels_with_gaps=False,
        minimum_length=0.9,
        minimum_interstation_distance_in_m=50_000,
        channel_priorities=["HDF","BDF","EDF","HHZ", "BHZ", "EHZ"],
    )
    mdl = MassDownloader(providers=[provider])
    mdl.download(cd, res, mseed_storage=name + "/waveforms", stationxml_storage=name + "/stations")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Download and process seismic data based on provided coordinates and time.')
    parser.add_argument('-lat', '--latitude', type=float, required=True, help='Latitude of the center point to grab data.')
    parser.add_argument('-lon', '--longitude', type=float, required=True, help='Longitude of the center point to grab data.')
    parser.add_argument('-s', '--starttime', type=lambda s: UTCDateTime(s), required=True, help='Start time in the format yyyy-MM-ddTHH:mm:ss')
    parser.add_argument('-l', '--length', type=float, nargs='?', default=None, help='Length of data to grab in seconds. If not provided, it will be calculated based on maxdistance and soundvelocity.')
    parser.add_argument('-d', '--maxdistance', type=float, nargs='?', default=100.0, help='Maximum distance of stations in kilometers. Default is 100 km.')
    parser.add_argument('-p', '--provider', type=str, nargs='?', default='IRIS', help='Data provider to grab data from. Default is IRIS.')
    parser.add_argument('-e', '--eventName', type=str, nargs='?', default=None, help='Name of the event to use for file names. If not provided, it will be generated from the start time.')
    parser.add_argument('-v', '--soundvelocity', type=float, nargs='?', default=.31, help='Sound velocity in meters per second. Default is 0.31 km/s.')
    parser.add_argument('-c', '--clean', type=bool, nargs='?', default=True, help='Remove traces without apparent triggers.')
    return parser.parse_args()

def get_distance(lat, lon, inv):
    """
    Calculate the distance between the event and each station.
    """
    distance = {}
    for network in inv:
        for station in network:
            distance[station.code] = haversine(lat, lon, station.latitude, station.longitude)
    return distance

def processData(st,inv,starttime,lp,hp,length,lat,lon,vel,height):
    def process_trace(trace, hp, lp, starttime, length, factor, distdict):
        trace = trace.copy()
        trace = trace.filter('bandpass', freqmin=hp, freqmax=lp, corners=2, zerophase=True)
        trace.trim(starttime=starttime - (length / 4.0), endtime=starttime + length, pad=True, fill_value=0)
        trace.normalize()
        times = trace.times()
        xtime = [(starttime - length / 4.0).datetime + timedelta(seconds=t) for t in times]
        data = {'x': xtime, 'y': trace.data * factor + distdict[trace.stats.station]}
        return ColumnDataSource(data=data)

    global distdict
    distdict = get_distance(lat, lon, inv)
    maxdist = max(distdict.values())
    factor = maxdist / len(st)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_trace, trace, hp, lp, starttime, length, factor, distdict) for trace in st]
        srcs = [future.result() for future in futures]

    dists = np.arange(0, maxdist, 0.5)
    # get ot from start button
    data2 = {'x': [starttime.datetime + timedelta(seconds=np.sqrt(dists[i]**2 + height**2) / vel) for i in range(len(dists))], 'y': dists}
    srcSonicBoom = ColumnDataSource(data=data2)
    return srcs, srcSonicBoom, distdict, maxdist, factor

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance between two points on the earth.
    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def lonlat_to_web_mercator(lon, lat):
    k = 6378137
    x = lon * (k * math.pi / 180.0)
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * k
    return x, y


## Main Porgram Starts Here
args = parse_arguments()
name = args.eventName or str(args.starttime).replace("T", "_").replace(":", "-")

if args.length is None:
    args.length = (1.5 * args.maxdistance) / args.soundvelocity

if os.path.isdir(name):
    st = read(name + "/waveforms/*.mseed")
    inv = read_inventory(name + "/stations/*.xml")
    print("Data found in " + name)
    print(st[0].stats)
else:
    print("Data not found, downloading")
    get_data(args.latitude, args.longitude, args.starttime, args.length, args.maxdistance, args.provider, name)
    st = read(name + "/waveforms/*.mseed")
    inv = read_inventory(name + "/stations/*.xml")

if(args.clean):
    toremove = []
    for i in range(len(st)):
        #do sta/lta
        try:
            cft = classic_sta_lta(st[i].data, int(20*st[i].stats.sampling_rate), int(100*st[i].stats.sampling_rate))
            if(max(cft) < 2.5):
                toremove.append(i)
        except:
            pass
    toremove.reverse()
    for i in toremove:
        st.pop(i)

#check if a F and Z component exisit for a given station name, and remove the Z if so
toremove = []
for i in range(len(st)):
    if('F' in st[i].stats.channel):
        for j in range(len(st)):
            if('Z' in st[j].stats.channel and st[i].stats.station == st[j].stats.station):
                toremove.append(j)
                break

toremove.reverse()
for i in toremove:
    st.pop(i)


st.resample(10.0)
st.merge(fill_value='interpolate')
st.detrend('demean')
st.detrend('linear')
st.taper(0.05)
stog = st.copy()
srcs, src2, distdict, maxdist, factor = processData(stog.copy(),inv,args.starttime,float(4.4),float(1.4),args.length,args.latitude,args.longitude, args.soundvelocity,10.)
print(src2)
recordSection = figure(height=700, width=700, x_axis_type="datetime")
recordSection.x_range.max_interval = timedelta(seconds=args.length*1.25)

picDict = {}

for i in range(len(srcs)):
    if(i==(len(srcs)-1)):
        recordSection.line(x='x', y='y', source=srcs[i], color='red', line_width=0.25,alpha=.5,name=str(st[i].get_id()))
    elif('HDF' in st[i].get_id() or 'BDF' in st[i].get_id() or 'EDF' in st[i].get_id()):
        recordSection.line(x='x', y='y', source=srcs[i], color='blue', line_width=0.25,alpha=.5,name=str(st[i].get_id()))
    else:
        recordSection.line(x='x', y='y', source=srcs[i], color='black', line_width=0.25,alpha=.5,name=str(st[i].get_id()))
  

        
recordSection.line(x='x', y='y', source=src2, color='green', line_width=2, alpha=0.5,name='SonicBoom')
recordSection.yaxis.axis_label = 'Distance (km)'
recordSection.xaxis.axis_label = 'Time'
recordSection.xaxis[0].formatter = DatetimeTickFormatter(days="%m/%d %H:%M", hours="%H:%M", minutes="%H:%M:%S", seconds="%Ss")
recordSection.xaxis.major_label_orientation = math.pi / 4
recordSection.xaxis.major_label_text_font_size = "10pt"
recordSection.yaxis.major_label_text_font_size = "10pt"
recordSection.xaxis.axis_label_text_font_size = "12pt"
recordSection.yaxis.axis_label_text_font_size = "12pt"
recordSection.add_tools(CrosshairTool())
recordSection.add_tools(HoverTool(tooltips=[("Time", "@x{%F %T}"), ("Station", "$name")], formatters={'@x': 'datetime'}))

recordSectionZoom = figure(height=200, width=700,  tools="xpan", toolbar_location=None,  x_axis_location="above",x_range=((args.starttime-15).datetime, (args.starttime+15).datetime),x_axis_type="datetime")


range_tool = RangeTool(x_range=recordSectionZoom.x_range, start_gesture="pan")
range_tool.overlay.fill_color = "teal"
range_tool.overlay.fill_alpha = 0.2

srcZoom = srcs[-1]
recordSectionZoom.line(x='x', y='y', source=srcZoom, color='red', line_width=0.25,alpha=.5,name=str(st[-1].get_id()))
recordSection.add_tools(range_tool)

lp = TextInput(value=str(4.4), title="Low Pass Filter (Hz):")
hp = TextInput(value=str(1.4), title="High Pass Filter (Hz):")
lat = TextInput(value=str(round(args.latitude, 3)), title="Latitude:")
lon = TextInput(value=str(round(args.longitude, 3)), title="Longitude:")
start = TextInput(value=str(args.starttime), title="Start Time:")
height = TextInput(value=str(round(10., 3)), title="Height (km):")
button = Button(label="Update", button_type="success")

button2 = Button(label="Locate", button_type="success")
button3 = Button(label="Beam", button_type="success")

boxwidth = TextInput(value=str(round(4.0, 3)), title="Horizontal Search Range (deg):")
boxstep = TextInput(value=str(round(0.1, 3)), title="Horizontal Search Step (deg):")
heightmin = TextInput(value=str(round(10., 3)), title="Min Elevation (km):")
heightmax = TextInput(value=str(round(25., 3)), title="Max Elevation (km):")
heightstep = TextInput(value=str(round(5., 3)), title="Elevation Step (km):")

#make a dictionary of station locations
stationLocDict = {}
for network in inv:
    for station in network:
        stationLocDict[station.code] = (station.latitude,station.longitude)

# range bounds supplied in web mercator coordinates
map = figure(x_axis_type="mercator", y_axis_type="mercator",height=900, width=900)
map.add_tile("CartoDB Positron", retina=True)
xs = []
ys = []

for network in inv:
        for station in network:
            #convert lat and lon to web mercator
            x,y = lonlat_to_web_mercator(station.longitude,station.latitude)
            xs.append(x)
            ys.append(y)

srcStation = ColumnDataSource(data={'x':xs,'y':ys})
map.scatter(x='x', y='y', source=srcStation, size=14, color='red',marker='triangle')


# on botton click replot data useing new parameteres in processData
def update():
    global distdict, maxdist, factor, st, inv, args, srcs, src2, picDict
    new_lp = float(lp.value)
    new_hp = float(hp.value)
    new_lat = float(lat.value)
    new_lon = float(lon.value)
    new_height = float(height.value)
    new_start = UTCDateTime(start.value)
    new_srcs, new_src2, distdict, maxdist, factor = processData(stog.copy(), inv, new_start,new_lp , new_hp, args.length, new_lat, new_lon, args.soundvelocity,new_height)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(srcs[i].data.update, new_srcs[i].data) for i in range(len(new_srcs))]
        for future in futures:
            future.result()
            
    src2.data.update(new_src2.data)
            
    #update pick locations
    i=0
    for key in picDict:
        time = picDict[key]
        xx = [time,time]
        yy = [distdict[key]-factor/2,distdict[key]+factor/2]
        srcpick = ColumnDataSource(data={'x':xx,'y':yy})
        picksrcs[i].data.update(srcpick.data)
        i = i+1
        
def locate():
    #get the parameters
    if(len(picDict) < 3):
        print("Need at least 3 picks to locate")
        return
            
    new_boxwidth = float(boxwidth.value)
    new_boxstep = float(boxstep.value)
    new_heightmin = float(heightmin.value)
    new_heightmax = float(heightmax.value)
    new_heightstep = float(heightstep.value)
    latT = float(lat.value)
    lonT = float(lon.value)
    ot = UTCDateTime(start.value)
    lats = np.arange(latT-new_boxwidth/2,latT+new_boxwidth/2,new_boxstep)
    lons = np.arange(lonT-new_boxwidth/2,lonT+new_boxwidth/2,new_boxstep)
    heights = np.arange(new_heightmin,new_heightmax,new_heightstep)
    rmsarray = np.zeros((len(lats),len(lons),len(heights)))
    
    def calculate_rms(params):
        i, j, k, lat_val, lon_val, height_val = params
        
        arrivalTimes = np.array([
            (picDict[station] / 1000. - ot.timestamp) - 
            np.sqrt(haversine(lat_val, lon_val, stationLocDict[station][0], stationLocDict[station][1])**2 + height_val**2) / args.soundvelocity
            for station in picDict
        ])
        
        timediff = arrivalTimes.mean()
        rms = np.sqrt(np.mean((arrivalTimes - timediff)**2))
        return i, j, k, rms, ot + timediff

    bestot = ot
    bestrms = float('inf')
    params = [(i, j, k, lat_val, lon_val, height_val) for i, lat_val in enumerate(lats) for j, lon_val in enumerate(lons) for k, height_val in enumerate(heights)]
    
    with Pool() as pool:
        results = pool.map(calculate_rms, params)
    
    for i, j, k, rms, bestot_candidate in results:
        rmsarray[i, j, k] = rms
        if rms < bestrms:
            bestrms = rms
            bestot = bestot_candidate
    
    i, j, k = np.unravel_index(np.argmin(rmsarray), rmsarray.shape)
    lat.value = str(round(lats[i],2))
    lon.value = str(round(lons[j],2))
    height.value = str(round(heights[k],2))
    start.value = bestot.strftime('%Y-%m-%dT%H:%M:%S')
    update()
    
    map.renderers = []
    map.add_tile("CartoDB Positron", retina=True)
    
    #plot stations
    xs = []
    ys = []
    for network in inv:
        for station in network:
            x,y = lonlat_to_web_mercator(station.longitude,station.latitude)
            xs.append(x)
            ys.append(y)
    srcStation = ColumnDataSource(data={'x':xs,'y':ys})
    map.scatter(x='x', y='y', source=srcStation, size=14, color='red',marker='triangle')
    
    rmsarray = rmsarray[:,:,k]
    xs = []
    ys = []
    colors = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            x, y = lonlat_to_web_mercator(lons[j], lats[i])
            xs.append(x)
            ys.append(y)
            colors.append(rmsarray[i,j])
    
    color_mapper = LinearColorMapper(palette=Inferno[256], low=max(colors), high=min(colors))
    #map.scatter(x='x', y='y', source=srcRMS, size=10.0, color={'field': 'colors', 'transform': color_mapper}, marker='circle', fill_alpha=0.4, name='rms')
    
    # plot rmsarray as an image
    map.image(image=[rmsarray], x=min(xs), y=min(ys), dw=max(xs)-min(xs), dh=max(ys)-min(ys), color_mapper=color_mapper,alpha=0.3)
    
    x, y = lonlat_to_web_mercator(float(lon.value), float(lat.value))
    srcBest = ColumnDataSource(data={'x':[x], 'y':[y]})
    map.scatter(x='x', y='y', source=srcBest, size=15., fill_color='white', line_color='black', marker='star')
    
    #zoom around srcRMS
    map.x_range.start = min(xs)
    map.x_range.end = max(xs)
    map.y_range.start = min(ys)
    map.y_range.end = max(ys)
    
    
def beam(): 
    new_boxwidth = float(boxwidth.value)
    new_boxstep = float(boxstep.value)
    new_heightmin = float(heightmin.value)
    new_heightmax = float(heightmax.value)
    new_heightstep = float(heightstep.value)
    latT = float(lat.value)
    lonT = float(lon.value)
    lats = np.arange(latT-new_boxwidth/2,latT+new_boxwidth/2,new_boxstep)
    lons = np.arange(lonT-new_boxwidth/2,lonT+new_boxwidth/2,new_boxstep)
    heights = np.arange(new_heightmin,new_heightmax,new_heightstep)
    beamarray = np.zeros((len(lats),len(lons),len(heights)))
    beamarrayIndex = np.zeros((len(lats),len(lons),len(heights)))
    
    #get sample rate
    stenv = stog.copy()
    stenv = stenv.filter('bandpass', freqmin=float(hp.value), freqmax=float(lp.value), corners=2, zerophase=True)
    sr = stenv[0].stats.sampling_rate
    stenv= stenv.normalize()
    
    remove = []
    for i in range(len(stenv)):
        stenv[i].data = signal.filter.envelope(stenv[i].data)
        stenv[i] = stenv[i].detrend('demean')
        stenv[i] = stenv[i].filter('highpass', freq=0.1, corners=2, zerophase=True)
        stenv[i].data = stenv[i].data / np.std(stenv[i].data)
        if(max(stenv[i].data) < 10.0):
            remove.append(i)
    remove.reverse()
    for i in remove:
        stenv.pop(i)

    #window for maximum given difference velocities
    timewindow = (1./.28)*111.12*new_boxstep - (1./.34)*111.12*new_boxstep
    
    times = stenv[0].times()   
        
    for i in range(len(lats)):
        for j in range(len(lons)):
            for k in range(len(heights)):
                shifted = np.zeros((len(stenv),len(times)))
                for l in range(len(stenv)):
                    #get staion location fron inv
                    slat = inv.get_coordinates(stenv[l].get_id())['latitude']
                    slon = inv.get_coordinates(stenv[l].get_id())['longitude']
                    dist = haversine(lats[i], lons[j], slat,slon)
                    time = np.sqrt(dist**2 + heights[k]**2) / args.soundvelocity
                    index = int(time * sr)
                    shifted[l] = maximum_filter1d(np.roll(stenv[l].data, -index),int(3*timewindow*sr))
                    
                stack = np.sum(shifted, axis=0)
                beamarray[i, j, k] = np.max(stack)
                beamarrayIndex[i, j, k] = np.argmax(stack)
         
    i, j, k = np.unravel_index(np.argmax(beamarray), beamarray.shape)
    lat.value = str(round(lats[i],2))
    lon.value = str(round(lons[j],2))
    height.value = str(round(heights[k],2))
    maxindex = int(beamarrayIndex[np.unravel_index(np.argmax(beamarray), beamarray.shape)])
    start.value = (args.starttime + timedelta(seconds=times[maxindex])).strftime('%Y-%m-%dT%H:%M:%S')
    update()
    
    map.renderers = []
    map.add_tile("CartoDB Positron", retina=True)
    
    #plot stations
    xs = []
    ys = []
    for network in inv:
        for station in network:
            x,y = lonlat_to_web_mercator(station.longitude,station.latitude)
            xs.append(x)
            ys.append(y)
    srcStation = ColumnDataSource(data={'x':xs,'y':ys})
    map.scatter(x='x', y='y', source=srcStation, size=14, color='red',marker='triangle')
    
    beamarray = beamarray[:,:,k]
    xs = []
    ys = []
    colors = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            x, y = lonlat_to_web_mercator(lons[j], lats[i])
            xs.append(x)
            ys.append(y)
            colors.append(beamarray[i,j])
    
    color_mapper = LinearColorMapper(palette=Inferno[256], low=max(colors), high=min(colors))
    
    # plot rmsarray as an image
    map.image(image=[beamarray], x=min(xs), y=min(ys), dw=max(xs)-min(xs), dh=max(ys)-min(ys), color_mapper=color_mapper,alpha=0.3)
    
    x, y = lonlat_to_web_mercator(float(lon.value), float(lat.value))
    srcBest = ColumnDataSource(data={'x':[x], 'y':[y]})
    map.scatter(x='x', y='y', source=srcBest, size=15., fill_color='white', line_color='black', marker='star')
    
    #zoom around srcRMS
    map.x_range.start = min(xs)
    map.x_range.end = max(xs)
    map.y_range.start = min(ys)
    map.y_range.end = max(ys)
    
button.on_click(update)
button2.on_click(locate)
button3.on_click(beam)

picksrcs = []
#one tap add to pickDict
def tap(event):
    if event.x < recordSection.x_range.end and event.x > recordSection.x_range.start and event.y < recordSection.y_range.end and event.y > recordSection.y_range.start:
        #get the name of the closest line
        closest = None
        closestDist = float('inf')
        for key in distdict:
            if abs(distdict[key] - event.y) < closestDist:
                closest = key
                closestDist = abs(distdict[key] - event.y)
        
        global picDict
        picDict[closest] = event.x
        #plot pick
        xx = [event.x, event.x]
        yy = [distdict[closest] - factor / 2, distdict[closest] + factor / 2]
        srcpick = ColumnDataSource(data={'x': xx, 'y': yy})
        picksrcs.append(srcpick)
        recordSection.line(x='x', y='y', source=picksrcs[-1], color='blue', line_width=2, alpha=1., name='pick')
        recordSectionZoom.line(x='x', y='y', source=picksrcs[-1], color='blue', line_width=2, alpha=1., name='pick')
        
        
selected = -1
name = None
def tapline(event):
    s = -1
    
    #remove picks from zoom plot
    for r in recordSectionZoom.renderers:
        if r.name == 'pick':
            recordSectionZoom.renderers.remove(r)   
            
    
    #find the closest line to the tap
    closest = None
    closestDist = float('inf')
    for key in distdict:
        if abs(distdict[key] - event.y) < closestDist:
            closest = key
            closestDist = abs(distdict[key] - event.y)    
    
    
    #loop through stream to get i
    for i in range(len(st)):
        if closest in st[i].get_id():
            s=i
            break
    
    name = closest
    
    # loop through renders and make them all black eccept for the selected
    for r in recordSection.renderers:
        #make sure not sonic boom or pick
        if(r.name != 'SonicBoom' and r.name != 'pick'):
            if r.name == str(st[s].get_id()):
                r.glyph.line_color = 'red'
            elif('HDF' in r.name or 'BDF' in r.name or 'EDF' in r.name):
                r.glyph.line_color = 'blue'
            else:
                r.glyph.line_color = 'black'
    
    #remvoe all renders from the zoomplo    
    for r in recordSectionZoom.renderers:
        recordSectionZoom.renderers.remove(r)
            
    #update the source of the line plotted in the zoom plot
    recordSectionZoom.line(x='x', y='y', source=srcs[s], color='red', line_width=0.25,alpha=1.,name=str(st[s].get_id()))
    
    #add a pick plot if there is a pick
    if closest in picDict:
        time = picDict[closest]
        xx = [time,time]
        yy = [distdict[closest]-factor/2,distdict[closest]+factor/2]
        srcpick = ColumnDataSource(data={'x':xx,'y':yy})
        recordSectionZoom.line(x='x', y='y', source=srcpick, color='blue', line_width=2, alpha=1.,name='pick')
    
    recordSectionZoom.y_range.start = distdict[closest] - factor
    recordSectionZoom.y_range.end = distdict[closest] + factor
     
    #update selected
    selected = s
    
    
recordSectionZoom.on_event(DoubleTap, tap)
recordSection.on_event(Tap, tapline)
inputs = column(button,lp,hp,lat,lon,start,height,button2,boxwidth,boxstep,heightmin,heightmax,heightstep,button3)
waveformsFig = column(recordSection,recordSectionZoom)
layout = row(waveformsFig,inputs,map)
doc = curdoc()
doc.add_root(layout)
