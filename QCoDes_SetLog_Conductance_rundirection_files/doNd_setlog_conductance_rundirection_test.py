from typing import Callable, Sequence, Union, Tuple, List, Optional
import os
import time
import datetime
import json
import ast 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText # to make text box
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

from qcodes.dataset.measurements import Measurement
from qcodes.instrument.base import _BaseParameter
from qcodes.dataset.plotting import plot_by_id
from qcodes import config
from qcodes.instrument.base import Instrument

from qdev_wrappers.dataset.export_functions import export_by_id, export_snapshot_by_id, export_by_id_pd

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
AxesTupleListWithRunId = Tuple[int, List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
number = Union[float, int]




datapath='C:\\Users\\felix\\Dropbox\\Informatik\\QCodes_Notebooks\\DataExport\\'



def do0d(*param_meas:  Union[_BaseParameter, Callable[[], None]],
         do_plot: bool = True) -> AxesTupleListWithRunId:
    """
    Perform a measurement of a single parameter. This is probably most
    useful for an ArrayParamter that already returns an array of data points

    Args:
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The run_id of the DataSet created
    """
    meas = Measurement()
    output = []

    for parameter in param_meas:
        meas.register_parameter(parameter)
        output.append([parameter, None])


    with meas.run() as datasaver:

        for i, parameter in enumerate(param_meas):
            if isinstance(parameter, _BaseParameter):
                output[i][1] = parameter.get()
            elif callable(parameter):
                parameter()
        datasaver.add_result(*output)
    dataid = datasaver.run_id

    if do_plot is True:
        ax, cbs = _save_image(datasaver)
    else:
        ax = None,
        cbs = None
    
    os.makedirs(datapath+'{}'.format(datasaver.run_id))
    inst=list(meas.parameters.values())
    exportpath=datapath+'{}'.format(datasaver.run_id)+'/{}_set_{}_set.dat'.format(inst[0].name,inst[1].name)
    exportsnapshot=datapath+'{}'.format(datasaver.run_id)+'/snapshot.txt'
    export_by_id(dataid,exportpath)
    export_snapshot_by_id(dataid,exportsnapshot)
    
    
    return dataid, ax, cbs


def do1d(param_set: _BaseParameter, start: number, stop: number,
         num_points: int, delay: number,
         *param_meas: Union[_BaseParameter, Callable[[], None]],
         enter_actions: Sequence[Callable[[], None]] = (),
         exit_actions: Sequence[Callable[[], None]] = (),
         do_plot: bool = True,
         do2dbuf: str = '',
         conDuct: Instrument = None) \
        -> AxesTupleListWithRunId:
    """
	adapted for logging settings by felix 17.04.2020
		-added argument do2buf
		-added _export_settings functionality

	adapted for live plotting of conductance by felix 17.04.2020
		-added argument conDuct
		-conDuct is a virtual parameter who has to be called as an optional argument in do1D.
		 conDuct has a function calcG() which allows to calculate the division of two given 
		 parameters or one parameter and a float number. See init file for more info.

    Perform a 1D scan of ``param_set`` from ``start`` to ``stop`` in
    ``num_points`` measuring param_meas at each step. In case param_meas is
    an ArrayParameter this is effectively a 2d scan.

    Args:
        param_set: The QCoDeS parameter to sweep over
        start: Starting point of sweep
        stop: End point of sweep
        num_points: Number of points in sweep
        delay: Delay after setting paramter before measurement is performed
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The run_id of the DataSet created
    """
    meas = Measurement()
    meas.register_parameter(
        param_set)  # register the first independent parameter
    output = []
    param_set.post_delay = delay
    interrupted = False

    for action in enter_actions:
        # this omits the posibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())
    for action in exit_actions:
        meas.add_after_run(action, ())

    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be
    # reimplemented from scratch
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter, setpoints=(param_set,))
            output.append([parameter, None])
        if conDuct != None:
            meas.register_parameter(conDuct.G, setpoints=(param_set,))
            output.append([conDuct.G, None])
    try:
        with meas.run() as datasaver:
            start_time = time.perf_counter()
            os.makedirs(datapath+'{}'.format(datasaver.run_id))   
            for set_point in np.linspace(start, stop, num_points):
                param_set.set(set_point)
                output = []
                for parameter in param_meas:
                    if isinstance(parameter, _BaseParameter):
                        output.append((parameter, parameter.get()))
                    elif callable(parameter):
                        parameter()
                if conDuct != None:
                    output.append((conDuct.G, conDuct.calcG(output)))
                    
                datasaver.add_result((param_set, set_point),
                                      *output)
    except KeyboardInterrupt:
        interrupted = True
        
    stop_time = time.perf_counter()
    
    dataid = datasaver.run_id  # convenient to have for plotting

    if interrupted:
        inst=list(meas.parameters.values())
        exportpath=datapath+'{}'.format(datasaver.run_id)+'/{}_set_{}_set.dat'.format(inst[0].name,inst[1].name)
        exportsnapshot=datapath+'{}'.format(datasaver.run_id)+'/snapshot.txt'
        #export_by_id(dataid,exportpath)
        export_by_id_pd(dataid,exportpath)
        export_snapshot_by_id(dataid,exportsnapshot)
        _export_settings(datasaver.run_id,inst,do2dbuf)
        stop_time = time.perf_counter()
        print("Acquisition took:  %s seconds " % (stop_time - start_time))
        raise KeyboardInterrupt

    print("Acquisition took:  %s seconds " % (stop_time - start_time))        
    inst=list(meas.parameters.values())
    exportpath=datapath+'{}'.format(datasaver.run_id)+'/{}_set_{}_set.dat'.format(inst[0].name,inst[1].name)
    exportsnapshot=datapath+'{}'.format(datasaver.run_id)+'/snapshot.txt'
    #export_by_id(dataid,exportpath)
    export_by_id_pd(dataid,exportpath)
    export_snapshot_by_id(dataid,exportsnapshot)
    #added by felix 05.03.2020 
    _export_settings(datasaver.run_id,inst,do2dbuf)
    
    if do_plot is True:
        ax, cbs = _save_image(datasaver, inst)
    else:
        ax = None,
        cbs = None
        
    return dataid, ax, cbs


def do2d(param_set1: _BaseParameter, start1: number, stop1: number,
         num_points1: int, delay1: number,
         param_set2: _BaseParameter, start2: number, stop2: number,
         num_points2: int, delay2: number,
         *param_meas: Union[_BaseParameter, Callable[[], None]],
         set_before_sweep: Optional[bool] = False,
         enter_actions: Sequence[Callable[[], None]] = (),
         exit_actions: Sequence[Callable[[], None]] = (),
         before_inner_actions: Sequence[Callable[[], None]] = (),
         after_inner_actions: Sequence[Callable[[], None]] = (),
         write_period: Optional[float] = None,
         flush_columns: bool = False,
         do_plot: bool=True,
         conDuct: Instrument = None) -> AxesTupleListWithRunId:

    """
    adapted for logging settings by felix 17.04.2020
		-added argument do2buf
		-added _export_settings functionality

	adapted for live plotting of conductance by felix 17.04.2020
		-added argument conDuct
		-conDuct is a virtual parameter who has to be called as an optional argument in do1D.
		 conDuct has a function calcG() which allows to calculate the division of two given 
		 parameters or one parameter and a float number. See init file for more info.

    Perform a 1D scan of ``param_set1`` from ``start1`` to ``stop1`` in
    ``num_points1`` and ``param_set2`` from ``start2`` to ``stop2`` in
    ``num_points2`` measuring param_meas at each step.

    Args:
        param_set1: The QCoDeS parameter to sweep over in the outer loop
        start1: Starting point of sweep in outer loop
        stop1: End point of sweep in the outer loop
        num_points1: Number of points to measure in the outer loop
        delay1: Delay after setting parameter in the outer loop
        param_set2: The QCoDeS parameter to sweep over in the inner loop
        start2: Starting point of sweep in inner loop
        stop2: End point of sweep in the inner loop
        num_points2: Number of points to measure in the inner loop
        delay2: Delay after setting paramter before measurement is performed
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        set_before_sweep: if True the outer parameter is set to its first value
            before the inner parameter is swept to its next value.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        before_inner_actions: Actions executed before each run of the inner loop
        after_inner_actions: Actions executed after each run of the inner loop
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The run_id of the DataSet created
    """

    meas = Measurement()
    if write_period:
        meas.write_period = write_period
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    interrupted = False
    for action in enter_actions:
        # this omits the possibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())

    for action in exit_actions:
        meas.add_after_run(action, ())

    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints=(param_set1, param_set2))
        if conDuct != None:
            meas.register_parameter(conDuct.G, setpoints=(param_set1, param_set2))

    try:
        with meas.run() as datasaver:
            start_time = time.perf_counter()
            os.makedirs(datapath+'{}'.format(datasaver.run_id))   
            for set_point1 in np.linspace(start1, stop1, num_points1):
                if set_before_sweep:
                    param_set2.set(start2)

                param_set1.set(set_point1)
                for action in before_inner_actions:
                    action()
                for set_point2 in np.linspace(start2, stop2, num_points2):
                    # skip first inner set point if `set_before_sweep`
                    if set_point2 == start2 and set_before_sweep:
                        pass
                    else:
                        param_set2.set(set_point2)
                    output = []
                    for parameter in param_meas:
                        if isinstance(parameter, _BaseParameter):
                            output.append((parameter, parameter.get()))
                        elif callable(parameter):
                            parameter()
                            
                        if conDuct != None:
                            output.append((conDuct.G, conDuct.calcG(output)))
                            
                    datasaver.add_result((param_set1, set_point1),
                                         (param_set2, set_point2),
                                         *output)
                for action in after_inner_actions:
                    action()
                if flush_columns:
                    datasaver.flush_data_to_database()
    
        stop_time = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True

    dataid = datasaver.run_id

    if interrupted:
        inst=list(meas.parameters.values())
        exportpath=datapath+'{}'.format(datasaver.run_id)+'/{}_set_{}_set.dat'.format(inst[0].name,inst[1].name)
        exportsnapshot=datapath+'{}'.format(datasaver.run_id)+'/snapshot.txt'
        #export_by_id(dataid,exportpath)
        export_by_id_pd(dataid,exportpath)
        export_snapshot_by_id(dataid,exportsnapshot)
        #added by felix 05.03.2020 
        _export_settings(datasaver.run_id,inst)

        stop_time = time.perf_counter()
        print("Acquisition took:  %s seconds " % (stop_time - start_time))
        raise KeyboardInterrupt

  
    inst=list(meas.parameters.values())
    exportpath=datapath+'{}'.format(datasaver.run_id)+'/{}_set_{}_set.dat'.format(inst[0].name,inst[1].name)
    exportsnapshot=datapath+'{}'.format(datasaver.run_id)+'/snapshot.txt'
    #export_by_id(dataid,exportpath)
    export_by_id_pd(dataid,exportpath)
    export_snapshot_by_id(dataid,exportsnapshot)
    # _export_settings(datasaver.run_id,inst)
    
    if do_plot is True:
        ax, cbs = _save_image(datasaver,inst)
    else:
        ax = None,
        cbs = None
        
    print("Acquisition took:  %s seconds " % (stop_time - start_time))

    return dataid, ax, cbs

def _save_image(datasaver,inst) -> AxesTupleList: 
    """
     adapted for showing little arrow for sweep direction by felix 24.04.2020
      -the arrows can be switched on and off by the bool add_drctn
    
    adapted for adding legend box to pngs by anders 17.04.2020
      -added argument inst: pass on swept parameters to get meta data
      -added argument metabox: bool to include box with metadata
      -the Legendbox can be switched on and off by the bool add_Label

    Save the plots created by datasaver as pdf and png

    Args:
        datasaver: a measurement datasaver that contains a dataset to be saved as plot.

    """
    plt.ioff()
    dataid = datasaver.run_id
    start = time.time()
    axes, cbs = plot_by_id(dataid)
    stop = time.time()
    print(f"plot by id took {stop-start}")

    mainfolder = config.user.mainfolder
    experiment_name = datasaver._dataset.exp_name
    sample_name = datasaver._dataset.sample_name 
    
    storage_dir = os.path.join(mainfolder, experiment_name, sample_name) 
    

    os.makedirs(storage_dir, exist_ok=True)

    png_dir = os.path.join(storage_dir, 'png')
    pdf_dif = os.path.join(storage_dir, 'pdf')

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dif, exist_ok=True)

    save_pdf = False
    save_png = True
    add_Label = False
    add_drctn = True
    
    # added 24.04.2020 by felix------------
    sgn = []
    if add_drctn:
        dataset = datasaver.dataset
        setParamList = dataset.get_parameters()
        for setParam in setParamList:
            if setParam.depends_on == '':
                if setParam.type == 'numeric':
                    axisData = dataset.get_data(setParam)
                    sgn.append(axisData[0][0]>=axisData[-1][0])
                if setParam.type == 'array':
                    axisData = dataset.get_data(setParam)
                    sgn.append(axisData[0][0][0]>=axisData[0][0][-1])

            
    # this block is modified by AK - make sure it works -------------------
    newDict = _export_settings(dataid, inst)
    newDict.pop('Parameter')
    meta_list = [key+' = '+str(newDict[key][0])+''+newDict[key][1] for key in newDict]

    for i, ax in enumerate(axes):
        # ---- new, create list structure to make box
        meta_string = ''
        for j,s in enumerate(meta_list):
            if j == len(meta_list)-1:
                meta_string += s
            else:
                meta_string += s+'\n'

        if save_pdf:
            full_path = os.path.join(pdf_dif, f'{dataid}_{i}.pdf')
            if len(meta_list)>0 and add_Label:
                ax.add_artist(AnchoredText(meta_string, loc=6, borderpad = -14))
            if add_drctn:
                xcor = [-3,-12]
                if sgn[0]: xcor=xcor[::-1]
                ax.annotate('', xy=(xcor[0],0), xytext=(xcor[1],0), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.8, headwidth = 3, headlength = 3),xycoords = 'axes points',textcoords = 'axes points')
                if len(sgn)==2:
                    if sgn[1]: xcor=xcor[::-1]
                    ax.annotate('', xy=(0,xcor[0]), xytext=(0,xcor[1]), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.8, headwidth = 3, headlength = 3),xycoords = 'axes points',textcoords = 'axes points')

            ax.figure.savefig(full_path, dpi=500)
            
        if save_png:
            full_path = os.path.join(png_dir, f'{dataid}_{i}.png')
            if len(meta_list)>0 and add_Label:
                ax.add_artist(AnchoredText(meta_string, loc=6, borderpad = -14))
            if add_drctn:
                xcor = [-3,-12]
                if sgn[0]: xcor=xcor[::-1]
                ax.annotate('', xy=(xcor[0],0), xytext=(xcor[1],0), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.8, headwidth = 3, headlength = 3),xycoords = 'axes points',textcoords = 'axes points')
                if len(sgn)==2:
                    if sgn[1]: xcor=xcor[::-1]
                    ax.annotate('', xy=(0,xcor[0]), xytext=(0,xcor[1]), arrowprops=dict(facecolor='black', shrink=0.05, width = 0.8, headwidth = 3, headlength = 3),xycoords = 'axes points',textcoords = 'axes points')

            ax.figure.tight_layout()
            ax.figure.savefig(full_path, dpi=500)
    #--------- end of modified section        
    plt.ion()
    return axes, cbs



def _export_settings(runID, inst, do2dbuf: str = ''):
    """
	added by felix 17.04.2020

    Outputs a json file which cotains certain settings of the station instruments. 
	This file is helpful for further data plotting and proccessing
    Args:
        runID: 	id of the current measurement which allows to read and write into
				the correct folder where the snapshot and rawdata is stored
        inst: 	List of all parameters registered for the measurement and the corresponding setpoint parameter
        do2dbuf: string which contains the second swept parameter in a do2dbuf measurement. do2dbuf calls do1d, 
        		so do1d needs to know whether it is called as individual function or as a part of do2dbuf.


    """
   #extract info from snapshot.txt
    newDict={}
    with open(datapath+'{}/'.format(runID) + 'snapshot.txt') as f:
        snap = f.read()
        snap = ast.literal_eval(snap)

        #QDAC
        try:
	        val=snap["station"]["instruments"]["qdac"]["parameters"]
	        for i in val.keys():
	            if val[i]["unit"]=='V' and val[i]["raw_value"] != 0:
	                newDict[i]=round(val[i]["raw_value"],3)
        except:
            pass

        #MAGNET
        try:
   	        val=snap["station"]["instruments"]["magnet"]["parameters"]
   	        for i in ['x_measured','y_measured','z_measured']:
   	            if val[i]["unit"]=='T' and val[i]["raw_value"] != 0:
   	                newDict[val[i]["label"]]=round(val[i]["raw_value"],3)
        except:
            pass

        #YOKO
        try:
	        val=snap["station"]["instruments"]["yoko"]["parameters"]
	        newDict[val["Vsd"]["label"]]=round(val["Vsd"]["raw_value"],4)
        except:
            pass

        #LOCKIN Amp
        try:
            val = snap["station"]["instruments"]["lockin_1"]["parameters"]["amplitude"]["value"]*10
            newDict["L_amp"]=[round(val), 'µV']
        except:
            pass
        
        #DummyStation
        try:
            newDict["Vd"]=[6, 'µV']
            newDict["Btest"]=[-9.4E-3, 'mT']
        except:
            pass

    ##Get names of the swept Parameters 
    aList=[]
    for i in inst:
        aList.append(i.name)

    if do2dbuf:
        aList.append(do2dbuf)

    newDict["Parameter"]={}
    for i in aList:
        newDict["Parameter"][i]=''
        if i in newDict.keys():
            newDict.pop(i)

    ##Output Data into json file as py dict
    exportpath=datapath+'{}'.format(runID)+'/plotSettings.json'
    with open(exportpath, 'w+') as ofi:
        json.dump(newDict, ofi)

    return newDict