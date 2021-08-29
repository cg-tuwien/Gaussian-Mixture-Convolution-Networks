from PIL import Image, ImageDraw
import numpy as np
import math

width = 1500
height = 1500

midx = width/2.0
midy = height/2.0
scale = 0.01

l1 = 2
l2 = 0.001#0.0001
eigenvec = np.array([[1, 1], [1, -1]])
eigenval = np.array([[l1, 0], [0, l2]])
cov = eigenvec @ eigenval @ np.linalg.inv(eigenvec)
cov_inv = np.linalg.inv(cov)

reg = 0#0.005
cov = cov + np.array([[reg, 0], [0, reg]])
cov_inv = np.linalg.inv(cov)

cm_data = [[2.62942888e-03,7.89377115e-04,6.66883520e-04],
           [4.11477266e-03,1.58885740e-03,1.75511247e-03],
           [5.84012537e-03,2.62348048e-03,3.33216897e-03],
           [7.80911062e-03,3.87878636e-03,5.37755505e-03],
           [1.00311687e-02,5.34043877e-03,7.88775080e-03],
           [1.25115136e-02,6.99663616e-03,1.08767011e-02],
           [1.52496834e-02,8.83860776e-03,1.43687876e-02],
           [1.82418242e-02,1.08599677e-02,1.83915638e-02],
           [2.14826541e-02,1.30557183e-02,2.29747276e-02],
           [2.49665429e-02,1.54218590e-02,2.81475649e-02],
           [2.86879173e-02,1.79549133e-02,3.39404435e-02],
           [3.26413668e-02,2.06518196e-02,4.03843180e-02],
           [3.68216090e-02,2.35099252e-02,4.70673872e-02],
           [4.12062443e-02,2.65266658e-02,5.37749094e-02],
           [4.55584261e-02,2.96998580e-02,6.05133310e-02],
           [4.98439535e-02,3.30275023e-02,6.72874217e-02],
           [5.40653394e-02,3.65077345e-02,7.41016781e-02],
           [5.82246968e-02,4.01387653e-02,8.09606843e-02],
           [6.23237743e-02,4.37792579e-02,8.78671386e-02],
           [6.63639834e-02,4.73690115e-02,9.48242245e-02],
           [7.03464191e-02,5.09155915e-02,1.01834851e-01],
           [7.42718806e-02,5.44217873e-02,1.08901689e-01],
           [7.81408887e-02,5.78901669e-02,1.16027198e-01],
           [8.19536989e-02,6.13231185e-02,1.23213643e-01],
           [8.57103132e-02,6.47228845e-02,1.30463105e-01],
           [8.94104879e-02,6.80915919e-02,1.37777487e-01],
           [9.30537391e-02,7.14312790e-02,1.45158514e-01],
           [9.66393462e-02,7.47439194e-02,1.52607730e-01],
           [1.00166327e-01,7.80313899e-02,1.60126845e-01],
           [1.03633420e-01,8.12954848e-02,1.67717781e-01],
           [1.07039238e-01,8.45382366e-02,1.75380667e-01],
           [1.10382054e-01,8.77615065e-02,1.83116641e-01],
           [1.13659758e-01,9.09670236e-02,1.90927741e-01],
           [1.16870272e-01,9.41570309e-02,1.98812642e-01],
           [1.20010757e-01,9.73331973e-02,2.06774144e-01],
           [1.23078566e-01,1.00497935e-01,2.14810277e-01],
           [1.26070343e-01,1.03653299e-01,2.22921975e-01],
           [1.28982438e-01,1.06801578e-01,2.31109132e-01],
           [1.31810971e-01,1.09945358e-01,2.39370217e-01],
           [1.34551498e-01,1.13087259e-01,2.47704283e-01],
           [1.37199120e-01,1.16230112e-01,2.56109858e-01],
           [1.39748450e-01,1.19376984e-01,2.64584854e-01],
           [1.42193578e-01,1.22531215e-01,2.73126484e-01],
           [1.44528037e-01,1.25696443e-01,2.81731154e-01],
           [1.46744767e-01,1.28876637e-01,2.90394351e-01],
           [1.48835549e-01,1.32075924e-01,2.99112455e-01],
           [1.50792253e-01,1.35299174e-01,3.07877566e-01],
           [1.52605120e-01,1.38551350e-01,3.16683676e-01],
           [1.54264257e-01,1.41838114e-01,3.25520945e-01],
           [1.55758214e-01,1.45165411e-01,3.34380083e-01],
           [1.57074815e-01,1.48539767e-01,3.43249244e-01],
           [1.58201154e-01,1.51968251e-01,3.52114024e-01],
           [1.59122999e-01,1.55458367e-01,3.60958709e-01],
           [1.59825170e-01,1.59018099e-01,3.69765057e-01],
           [1.60292236e-01,1.62655858e-01,3.78511034e-01],
           [1.60507086e-01,1.66380286e-01,3.87173619e-01],
           [1.60453178e-01,1.70200169e-01,3.95725054e-01],
           [1.60112923e-01,1.74124183e-01,4.04136124e-01],
           [1.59469606e-01,1.78160592e-01,4.12373974e-01],
           [1.58507587e-01,1.82316858e-01,4.20403239e-01],
           [1.57212551e-01,1.86599247e-01,4.28187142e-01],
           [1.55573483e-01,1.91012229e-01,4.35687323e-01],
           [1.53582542e-01,1.95558086e-01,4.42866125e-01],
           [1.51235765e-01,2.00236533e-01,4.49687983e-01],
           [1.48535235e-01,2.05044163e-01,4.56120221e-01],
           [1.45487936e-01,2.09974568e-01,4.62135721e-01],
           [1.42106639e-01,2.15018318e-01,4.67713886e-01],
           [1.38409632e-01,2.20163295e-01,4.72841758e-01],
           [1.34419647e-01,2.25395340e-01,4.77514629e-01],
           [1.30163640e-01,2.30698816e-01,4.81735812e-01],
           [1.25670666e-01,2.36057601e-01,4.85516095e-01],
           [1.20972005e-01,2.41455567e-01,4.88872526e-01],
           [1.16099603e-01,2.46877349e-01,4.91827118e-01],
           [1.11085756e-01,2.52308739e-01,4.94405432e-01],
           [1.05961976e-01,2.57737145e-01,4.96635114e-01],
           [1.00760163e-01,2.63151491e-01,4.98544881e-01],
           [9.55120248e-02,2.68542396e-01,5.00163510e-01],
           [9.02492816e-02,2.73902142e-01,5.01519064e-01],
           [8.50041003e-02,2.79224565e-01,5.02638337e-01],
           [7.98096374e-02,2.84504914e-01,5.03546481e-01],
           [7.47007522e-02,2.89739678e-01,5.04266785e-01],
           [6.97177762e-02,2.94925968e-01,5.04821484e-01],
           [6.48999841e-02,3.00062490e-01,5.05229624e-01],
           [6.02958761e-02,3.05147951e-01,5.05509934e-01],
           [5.59550410e-02,3.10182205e-01,5.05678319e-01],
           [5.19332248e-02,3.15165418e-01,5.05749468e-01],
           [4.82907314e-02,3.20098125e-01,5.05736669e-01],
           [4.50905520e-02,3.24981211e-01,5.05651798e-01],
           [4.23957320e-02,3.29815825e-01,5.05505436e-01],
           [4.02626480e-02,3.34603328e-01,5.05306986e-01],
           [3.87435108e-02,3.39345177e-01,5.05064983e-01],
           [3.78953292e-02,3.44042858e-01,5.04787330e-01],
           [3.77098674e-02,3.48698246e-01,5.04479997e-01],
           [3.81813040e-02,3.53312891e-01,5.04149533e-01],
           [3.93007590e-02,3.57888659e-01,5.03800749e-01],
           [4.10478299e-02,3.62427193e-01,5.03438752e-01],
           [4.33393374e-02,3.66930402e-01,5.03066998e-01],
           [4.61252725e-02,3.71399867e-01,5.02689792e-01],
           [4.93412780e-02,3.75837402e-01,5.02309891e-01],
           [5.29248683e-02,3.80244728e-01,5.01929896e-01],
           [5.68179217e-02,3.84623493e-01,5.01552223e-01],
           [6.09679428e-02,3.88975319e-01,5.01178924e-01],
           [6.53287681e-02,3.93301780e-01,5.00811778e-01],
           [6.98604359e-02,3.97604432e-01,5.00452164e-01],
           [7.45288605e-02,4.01884804e-01,5.00101108e-01],
           [7.93053638e-02,4.06144359e-01,4.99759507e-01],
           [8.41659634e-02,4.10384498e-01,4.99428145e-01],
           [8.90905809e-02,4.14606606e-01,4.99107448e-01],
           [9.40625259e-02,4.18812022e-01,4.98797616e-01],
           [9.90679853e-02,4.23002039e-01,4.98498706e-01],
           [1.04095542e-01,4.27177901e-01,4.98210603e-01],
           [1.09135783e-01,4.31340811e-01,4.97933032e-01],
           [1.14181018e-01,4.35491890e-01,4.97665839e-01],
           [1.19224893e-01,4.39632257e-01,4.97408421e-01],
           [1.24262229e-01,4.43763008e-01,4.97159864e-01],
           [1.29288890e-01,4.47885158e-01,4.96919411e-01],
           [1.34301606e-01,4.51999637e-01,4.96686553e-01],
           [1.39297764e-01,4.56107440e-01,4.96459726e-01],
           [1.44275427e-01,4.60209434e-01,4.96237972e-01],
           [1.49233185e-01,4.64306446e-01,4.96020209e-01],
           [1.54170095e-01,4.68399321e-01,4.95804670e-01],
           [1.59085634e-01,4.72488759e-01,4.95590439e-01],
           [1.63979655e-01,4.76575522e-01,4.95375480e-01],
           [1.68852330e-01,4.80660241e-01,4.95158499e-01],
           [1.73704148e-01,4.84743557e-01,4.94937578e-01],
           [1.78535858e-01,4.88826027e-01,4.94711074e-01],
           [1.83348475e-01,4.92908186e-01,4.94477034e-01],
           [1.88143237e-01,4.96990504e-01,4.94233635e-01],
           [1.92921613e-01,5.01073419e-01,4.93978789e-01],
           [1.97685262e-01,5.05157299e-01,4.93710616e-01],
           [2.02436060e-01,5.09242491e-01,4.93426841e-01],
           [2.07176032e-01,5.13329257e-01,4.93125601e-01],
           [2.11907419e-01,5.17417852e-01,4.92804464e-01],
           [2.16632578e-01,5.21508434e-01,4.92461523e-01],
           [2.21354065e-01,5.25601152e-01,4.92094360e-01],
           [2.26074552e-01,5.29696080e-01,4.91700870e-01],
           [2.30796860e-01,5.33793249e-01,4.91278808e-01],
           [2.35523942e-01,5.37892642e-01,4.90825903e-01],
           [2.40258854e-01,5.41994185e-01,4.90340032e-01],
           [2.45004783e-01,5.46097763e-01,4.89818895e-01],
           [2.49765000e-01,5.50203207e-01,4.89260359e-01],
           [2.54542875e-01,5.54310299e-01,4.88662246e-01],
           [2.59341854e-01,5.58418776e-01,4.88022443e-01],
           [2.64165459e-01,5.62528328e-01,4.87338837e-01],
           [2.69017257e-01,5.66638594e-01,4.86609448e-01],
           [2.73900888e-01,5.70749175e-01,4.85832215e-01],
           [2.78820003e-01,5.74859623e-01,4.85005279e-01],
           [2.83778295e-01,5.78969447e-01,4.84126782e-01],
           [2.88779488e-01,5.83078121e-01,4.83194847e-01],
           [2.93827255e-01,5.87185062e-01,4.82207974e-01],
           [2.98925383e-01,5.91289674e-01,4.81164188e-01],
           [3.04077488e-01,5.95391289e-01,4.80062335e-01],
           [3.09287324e-01,5.99489234e-01,4.78900647e-01],
           [3.14558482e-01,6.03582773e-01,4.77678022e-01],
           [3.19894623e-01,6.07671151e-01,4.76393034e-01],
           [3.25299289e-01,6.11753569e-01,4.75044661e-01],
           [3.30776032e-01,6.15829198e-01,4.73631789e-01],
           [3.36328301e-01,6.19897170e-01,4.72153595e-01],
           [3.41959561e-01,6.23956588e-01,4.70609148e-01],
           [3.47673089e-01,6.28006513e-01,4.68998066e-01],
           [3.53472355e-01,6.32045981e-01,4.67319341e-01],
           [3.59360362e-01,6.36073979e-01,4.65573315e-01],
           [3.65340641e-01,6.40089469e-01,4.63758835e-01],
           [3.71416070e-01,6.44091366e-01,4.61876581e-01],
           [3.77589740e-01,6.48078550e-01,4.59926667e-01],
           [3.83865063e-01,6.52049845e-01,4.57908542e-01],
           [3.90244726e-01,6.56004047e-01,4.55823565e-01],
           [3.96731753e-01,6.59939895e-01,4.53672414e-01],
           [4.03329311e-01,6.63856061e-01,4.51455688e-01],
           [4.10040517e-01,6.67751156e-01,4.49174421e-01],
           [4.16868194e-01,6.71623746e-01,4.46830611e-01],
           [4.23815433e-01,6.75472309e-01,4.44426098e-01],
           [4.30885376e-01,6.79295240e-01,4.41963130e-01],
           [4.38081233e-01,6.83090839e-01,4.39444422e-01],
           [4.45406299e-01,6.86857305e-01,4.36873253e-01],
           [4.52863960e-01,6.90592722e-01,4.34253569e-01],
           [4.60457701e-01,6.94295047e-01,4.31590113e-01],
           [4.68191289e-01,6.97962073e-01,4.28888307e-01],
           [4.76069265e-01,7.01591343e-01,4.26153686e-01],
           [4.84094844e-01,7.05180417e-01,4.23395563e-01],
           [4.92272918e-01,7.08726441e-01,4.20622425e-01],
           [5.00608132e-01,7.12226384e-01,4.17845446e-01],
           [5.09105680e-01,7.15676889e-01,4.15077677e-01],
           [5.17770224e-01,7.19074424e-01,4.12336091e-01],
           [5.26607596e-01,7.22414949e-01,4.09639812e-01],
           [5.35623627e-01,7.25694067e-01,4.07012582e-01],
           [5.44823822e-01,7.28907038e-01,4.04484047e-01],
           [5.54212845e-01,7.32048843e-01,4.02091350e-01],
           [5.63794947e-01,7.35114051e-01,3.99879834e-01],
           [5.73572411e-01,7.38097092e-01,3.97906169e-01],
           [5.83544270e-01,7.40992539e-01,3.96240946e-01],
           [5.93702258e-01,7.43796130e-01,3.94973157e-01],
           [6.04030456e-01,7.46505044e-01,3.94210080e-01],
           [6.14495930e-01,7.49120688e-01,3.94081609e-01],
           [6.25047603e-01,7.51649806e-01,3.94733167e-01],
           [6.35607758e-01,7.54108041e-01,3.96316252e-01],
           [6.46075090e-01,7.56520810e-01,3.98964276e-01],
           [6.56333572e-01,7.58922439e-01,4.02763276e-01],
           [6.66271923e-01,7.61351295e-01,4.07727094e-01],
           [6.75805154e-01,7.63842627e-01,4.13792783e-01],
           [6.84885689e-01,7.66423212e-01,4.20838761e-01],
           [6.93504538e-01,7.69108505e-01,4.28716374e-01],
           [7.01679216e-01,7.71904754e-01,4.37275496e-01],
           [7.09444052e-01,7.74811008e-01,4.46382612e-01],
           [7.16838782e-01,7.77822753e-01,4.55924323e-01],
           [7.23903743e-01,7.80933569e-01,4.65808021e-01],
           [7.30678403e-01,7.84135602e-01,4.75963421e-01],
           [7.37197120e-01,7.87421699e-01,4.86332939e-01],
           [7.43490693e-01,7.90784886e-01,4.96872488e-01],
           [7.49585475e-01,7.94219114e-01,5.07545456e-01],
           [7.55504923e-01,7.97718616e-01,5.18324639e-01],
           [7.61269690e-01,8.01278018e-01,5.29190401e-01],
           [7.66895871e-01,8.04893939e-01,5.40116837e-01],
           [7.72400310e-01,8.08561295e-01,5.51098023e-01],
           [7.77795891e-01,8.12276979e-01,5.62119337e-01],
           [7.83093992e-01,8.16038334e-01,5.73167166e-01],
           [7.88305524e-01,8.19842200e-01,5.84236899e-01],
           [7.93439693e-01,8.23686233e-01,5.95320882e-01],
           [7.98504630e-01,8.27568529e-01,6.06410600e-01],
           [8.03508069e-01,8.31486828e-01,6.17504101e-01],
           [8.08456829e-01,8.35439235e-01,6.28598743e-01],
           [8.13356785e-01,8.39424715e-01,6.39685171e-01],
           [8.18213764e-01,8.43441474e-01,6.50764013e-01],
           [8.23032872e-01,8.47488442e-01,6.61829943e-01],
           [8.27818989e-01,8.51564151e-01,6.72883601e-01],
           [8.32576485e-01,8.55668177e-01,6.83915081e-01],
           [8.37309682e-01,8.59798902e-01,6.94929320e-01],
           [8.42022524e-01,8.63955536e-01,7.05922771e-01],
           [8.46718831e-01,8.68137557e-01,7.16889310e-01],
           [8.51402293e-01,8.72344081e-01,7.27827765e-01],
           [8.56076465e-01,8.76574206e-01,7.38737735e-01],
           [8.60744853e-01,8.80827236e-01,7.49616584e-01],
           [8.65410947e-01,8.85102479e-01,7.60461756e-01],
           [8.70078253e-01,8.89399238e-01,7.71270715e-01],
           [8.74750339e-01,8.93716799e-01,7.82040871e-01],
           [8.79430883e-01,8.98054422e-01,7.92769507e-01],
           [8.84123727e-01,9.02411328e-01,8.03453685e-01],
           [8.88832967e-01,9.06786762e-01,8.14089214e-01],
           [8.93563065e-01,9.11180107e-01,8.24668996e-01],
           [8.98318646e-01,9.15590106e-01,8.35192237e-01],
           [9.03104891e-01,9.20015698e-01,8.45653811e-01],
           [9.07927526e-01,9.24455724e-01,8.56047480e-01],
           [9.12793000e-01,9.28909501e-01,8.66359346e-01],
           [9.17708142e-01,9.33375428e-01,8.76583703e-01],
           [9.22680429e-01,9.37852095e-01,8.86710427e-01],
           [9.27717666e-01,9.42339100e-01,8.96717987e-01],
           [9.32827237e-01,9.46835665e-01,9.06590694e-01],
           [9.38015096e-01,9.51342094e-01,9.16307695e-01],
           [9.43283190e-01,9.55861710e-01,9.25833847e-01],
           [9.48627160e-01,9.60398765e-01,9.35149450e-01],
           [9.54030897e-01,9.64963629e-01,9.44219731e-01],
           [9.59464808e-01,9.69569318e-01,9.53035846e-01],
           [9.64884463e-01,9.74233679e-01,9.61598376e-01],
           [9.70239259e-01,9.78972847e-01,9.69948034e-01],
           [9.75484299e-01,9.83798809e-01,9.78143763e-01],
           [9.80593547e-01,9.88713419e-01,9.86270418e-01]]

img = Image.new('RGB', (width, height), color='black')
pixelMap = img.load()

# max = 0.15
max = 0.4
min = 0.0

for x in range(width):
    for y in range(height):
        xr = (x - midx) * scale
        yr = (y - midy) * scale
        # density = 1.0 / (2 * np.pi * sx * sy) * np.exp(-0.5 * ((xr**2)/sx + (yr**2)/sy))
        pos = np.array([xr, yr])
        density = np.exp(-0.5 * pos.transpose() @ cov_inv @ pos) / np.sqrt((2*np.pi)**2 * np.linalg.det(cov))
        level = (density - min) / (max - min)
        if level > 1:
            level = 1
        if level < 0:
            level = 0
        level = 1 - level
        mapped = cm_data[math.floor(level * (len(cm_data)-1))]
        color = [int(255*x) for x in mapped]
        # dist = np.sqrt(xr**2 + yr**2)
        # bright = 255-int(dist*255)
        pixelMap[x,y] = (color[0], color[1], color[2])

# drawer = ImageDraw.Draw(img)
# evec1 = np.array([1, 1], dtype=np.float)
# evec1 /= np.linalg.norm(evec1)
# evec1 *= l1/scale
# evec2 = np.array([1, -1], dtype=np.float)
# evec2 /= np.linalg.norm(evec2)
# evec2 *= l2/scale
# drawer.line([midx, midy, midx + evec1[0], midy + evec1[1]], fill="blue", width=2)
# drawer.line([midx, midy, midx + evec2[0], midy + evec2[1]], fill="blue", width=2)

img.show()
img.save(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\graphics\gROT_l1" + str(l1).replace(".", "-") + "_l2" + str(l2).replace(".", "-") +
        #"_REG" + str(reg).replace(".", "-") +
        #"_EV" +
        "_s" + str(scale).replace(".", "-") + "_mx" + str(max).replace(".", "-") + "w.PNG")

#img.save(r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\thesis\graphics\gROT-REG" + str(reg).replace(".", "-")+ "_s" +
#         str(scale).replace(".", "-") + "_mx" + str(max).replace(".", "-") + "w.PNG")