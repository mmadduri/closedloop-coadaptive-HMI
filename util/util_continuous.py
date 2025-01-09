import numpy as np
import matplotlib 
## for figures
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

n_ch = 64
n_targ = 10
n_blocks = 2
n_conds = 8
keys = ['METACPHS_S106', 'METACPHS_S107','METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']
n_keys = len(keys)

# time variables
RAMP = 350 # samples - first 5 seconds
min_time = 20770 # 20770 for ~5 minutes
trial_time = 300 # 300 second trials
tscale = min_time/trial_time # number of samples per seconds
# indices of decoder updates - calculated in initial CPHS/fullanalysis notebook
# /Users/mmadduri/Documents/GitHub/meta-analysis/ContinuousDecoder/fullanalysis/20220321-fullanalysis-cphspaper-figures.ipynb
update_ix = np.array([0, 1200, 2402, 3604, 4806, 6008, 7210, 8412, 9614, 10816, 12018, 13220,
 14422, 15624, 16826, 18028, 19230, 20432, 20769])

# scaling from computer x,y to cm x,y
# ref_x*x_cm_to_au = ref_x in cm 
x_cm_to_au = 0.5110354085567264 
y_cm_to_au = 0.5085593585551849


# set colors
colors_targ = np.zeros((n_targ, 4))
for ii in range(n_targ):
    colors_targ[ii] = matplotlib.cm.viridis(ii / 8)

subj_labels = [str(i + 1) for i in range(n_keys)]
# for key in keys:
#     subj_labels.append(key[-3:])

# learning rate conditions
# slow --> alpha = 0.75 bc D_next = alpha*D_old + (1-alpha)*D_new, so more alpha = more D old
slow = [4, 5, 6, 7] 
fast = [0, 1, 2, 3]

# initialization conditions
pos_init = [0, 1, 4, 5]
neg_init = [2, 3, 6, 7]

# penalty parameter conditions
pD_3 = [0, 2, 4, 6] # pD = 1e-3
pD_4 = [1, 3, 5, 7] # pD = 1e-4

colors = dict()

colors['D'] = '#F1A340' # decoder
colors['target'] =  '#C00000'
colors['cursor'] = '#1A70E5'

# comparison of early error vs late error
colors['early'] = '#767171'
colors['late'] = '#262626'

# block 1 vs block 2
colors['B1'] = '#767171'
colors['B2'] = '#262626'

colors['emg'] = '#262626'

# encoder
colors['FF'] = '#9E527C'
colors['FB'] = '#316D9E'
colors['E'] = '#604FA5' # encoder/user


# decoder conditions
colors['slow'] = '#5d600f' #'#9dc9ae'
colors['fast'] =  '#9b9e48'  #'#38734e'
colors['pos_init'] = '#F5C6A6' #BDBDBD' #'paleturquoise'
colors['neg_init'] = '#d35805' #2d6899'#676767' #'cadetblue'
colors['pD_3'] = '#78b0ab' #'#1c364d' #'steelblue'
colors['pD_4'] = '#0c564f' #'steelblue'#'lightsteelblue'

# set figure size
fw = 12
fh = 8
lbl_size = 15
tck_size = 15

# use LaTeX to render symbols
matplotlib.rc('text',usetex=False)
font = { #'family' : 'sans-serif',
        'weight' : 'ultralight',
        'size'   : 14}
matplotlib.rc('font', **font)
rcParams['axes.titlesize'] = 5
rcParams['axes.labelsize'] = 6
rcParams['mathtext.fontset'] = 'cm'
rcParams["figure.dpi"] = 300.0

# Set rcParams to move ticks inside the plot
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

sns_custom_params = {"axes.spines.right": False, 
                     "axes.spines.top": False, 
                     "xtick.direction": "in", 
                     "ytick.direction": "in", 
                     'axes.linewidth': 1.0,
                     'xtick.major.width': 1.0, 
                     'ytick.major.width': 1.0,
                     'xtick.major.size': 3, 
                     'ytick.major.size': 3, 
                     'font.family': 'sans-serif',
                     'font.sans-serif': 'DejaVu Sans'}