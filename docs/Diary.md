# 📜 Diary

## 01/02/2023

- Reading Barnabé Deforet's internship report
  - Tool to generate dendograms: AstroDendro
- Meeting with Julien Montillaud
  - Defined global strategy
  - **Step 1**: Search existing tools to implement 3D convolutive neural network
    - Found [CIANNA](https://github.com/Deyht/CIANNA) by David Cornu
    - Found Tensorflow.Keras.layers.Conv3D
  - **Step 2**: Generate simplistic models
  - **Step 3**: Implement 3D convolutive neural network
  - **Step 4**: Once it works, try to increase the complexity of the dataset to be more realistic (complexe shape, noise etc.)
  - **Step 5**: Implement [dropout](https://inside-machinelearning.com/le-dropout-cest-quoi-deep-learning-explication-rapide/) to have statistical results
  - **Step 6**: Use the model on real data

## 02/02/2023

- Created project flowchart to have a global vision of the project:

<div align=center>

![](img/project_flowchart.png)

</div>

- Having trouble with AMUSE installation... I spend the entire day on it, but I finally succeeded to install it.
  <details>
  <summary>Installation problems</summary> 

  There was some additional dependencies to install and problem with deprecation of some functions used to build some modules, and after that there was still a problem that required to totally clear the cache of pip and reinstall all modules, so I don't really know which manipulation were usefull and which were not.

  </details>

## 03/02/2023

- Getting familiar with Barnabé's code
- I need an access to the computation cluster in order to run Barnabé's code... waiting for it.
- Continuing reading Barnabé Deforet's internship report and articles about LOC and SOC

## 07/02/2023

- Reading Keras documentation and try to have a first rough neural network that work with any shapes of data (3D but with any size on each axis) for input and output
  - **Vocabulary: Batch size**: Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
  - **Vocabulary: Channel**: A channel is a feature map. It is the last dimension in the data. For instance, in a grayscale image, the channel is 1, and in an RGB image, it is 3. In general, the channels axis is -1 for TensorFlow and Theano, and 1 for CNTK.
    - **What's the différence between 2D convolution on N channels and a 3D convolution on 1 channel?** The 2D convolution will treat all the channels independently, while the 3D convolution will treat all the channels together, so it will be more accurate to find schemes in the channel axis. Here, we want to extract relevant information from the difference between frequencies, so a 3D convolution looks a priori more relevant.
  - **Layer: MaxPooling3D**: Downsamples the input along its dimensions (depth, height, width) by taking the maximum value over the window defined by pool_size for each dimension along the features axis. The window is shifted by strides in each dimension. The resulting output when using "valid" padding option has a shape of:
    - output_shape = (input_shape - pool_size + 1) / strides
  - Got an idea of what to do while waiting to be able to generate huge physical simulations: make simplistic prestellar cores by using a gaussian repartition of the density and apply a plummer model on it to have a prestellar core base, then make it evolve a bit with some initial speed parameters before generating mock observations of it by simulating an artificial red shift. I will start this tomorrow.
  - Made the flowchart for this part:

<div align=center>

![](img/rough_prestellar_core_flowchart.png)

</div>

## 08/02/2023

- I did the simplistic prestellar core generation by using a slight different process than the original idea, here is the new flowchart:
  
<div align=center>

![](img/rough_mock_obs_flowchart.png)

</div>

- There is still some verifications and probably improvements to ensure this model is not totally absurd before generating a huge dataset.
- Also, there is a problem concerning the view axis. I avoid heavy computation to make a true observation from a distant point of view by summing the 3D datacube along a dimension. But according to the dimension on which I sum, I need to rotate the input cubes accordingly to the point of view before saving them. As it is easier to say but tricky to do, I decided to use only one point of view. Thus, the orientation of the cubes has no importance because the AI will try to match a cube without considering the physical orientation of the system represented inside. With only one point of view, I'm sure that the relative orientation between the obervation and the output will be the same, and then avoid convergence issues.

## 09/02/2023

- Added normalization at several steps of the process to have better control of the output.
- Removed the central symtery by generating the initial lorentzian distribution in a bigger space and cropping this space after the fourier transform.
- Got access to the computation cluster and started to got familiar with it.
- Generated a 1000 items dataset
- Facing a problem of memory when I try to train the neural network do to the huge size of the NN hyperparameters (more than 68 Billions for inner dense layers). The problem seems that I need one neuron for each pixel or voxel, which becomes huges even for image or spaces of 64 pixels/vocal wide. Thus, I need to find another solution in literature that doesn't involve dense layers, or at least that solve the problem in some way.

## 10/02/2023

- Found a solution to train a classic neural network model, taking adventage of the local property of the problem. In fact, an output column depend only on the corresponding input column (and maybe some neighbors columns) and not on the entire datacube. So I can train a neural network that will be focused on a small windows of columns and will scan the datacube to get all output columns and rebuild the output datacube. Here is the flowchart of the idea:

<div align=center>

![](img/local_neural_network_flowchart.png)

</div>

- Searched and found a solution to use Jupyter in VSC with a kernel hosted a computation node.
- Edited the function to generate the input/output vectors composed of columns insteade of entire images, but I'm facing a memory leak issue...
- Just thought to another problem: until now, I avoided the problem of orientation faced in the last part of the cloud generation process (see 08/02/2023). But as I'm now considering column (on the Z axis) at a specific X and Y coordinate, I need to ensure that all the axis are the same. Otherwise, I ask the AI to find the velocity and density profile along a line on X or Y (perpendicular to the line of sight) according to a spectrum for one given line of sight.

# 13/02/2023

- I think I solved the problem of orientation with an unconventional method. I will finish to implement it and test it tomorrow
  <details>
  <summary>Few details of the problem and method to solve it</summary>
  
  **The problem**: I have 3 cubes representing velocities along X, Y and Z axis for each space element (dimensions of the cubes correspond to space dimensions). The basis of Vx, Vy and Vz is classically oriented, but the space basis, guided by the index of the numpy array, is totally upside down. Moreover, I need a Vx', Vy' and Vz' in the basis of the observer... so that was a real 9 dimensionnale puzzle to solve.

  **The unconventional method**: As I need to deal with 2 basis seen from a certain point of view, I first tryind to draw the problem... but without success. THen I tried with basis made with post-its, but no success neither. I needed to modelize the problem. As I don't master any 3D software to modelize it, I used the only kind-of 3D software I know: Minecraft, on wich I represented physically the basis used by numpy and the basis of the velocity vectors with their respective orientation. With numpy I designed a simple data cube conatining only 1 on (x,0,0), 2 on (0,y,0) and 3 on (0,0,z) and I sum this cube successively over each observation axis. By ploting the result, I was able to determine the orientation of the observer. After that, I put the player view in the corresponding orientation and I was able to see how Vx', Vy' and Vz' depend on Vx, Vy and Vz for each oservation axis.

  ![](img/2023-02-13-17-01-42.png)

  **The result:** It will not be very explicit, but I got:

  - For an observer on X+:
    - Vx'(x',y',z') = -Vz(y,z,x)
    - Vy'(x',y',z') = -Vy(y,z,x)
    - Vz'(x',y',z') = Vy(y,z,x)
  - For an observer on Y+:
    - Vx'(x',y',z') = Vy(x,z,-y)
    - Vy'(x',y',z') = -Vx(x,z,-y)
    - Vz'(x',y',z') = Vz(x,z,-y)
  - For an observer on Z+:
    - Vx'(x',y',z') = Vy(x,y,z)
    - Vy'(x',y',z') = -Vz(x,y,z)
    - Vz'(x',y',z') = -Vx(x,y,z)

  For the observer on negative axis, it's the same but with the reversed x' and z' coordinates.

  </details>

- Discussed with Julien Montillaud and having now 2 new methods using NN to solve the problem which:
  - The one I considered until now, consisting of scanning the input data cube with a small windows to get only the data of a line of sight as a result and reconstruct progressively the output data cube. I will call it the "Scanner" method from now on.
  - Taking a very low resolution input cube, and then zooming on a child cube with always the same resolution. I will call it the "NCIS zoom" method from now on.
  - Taking the overhaul input cube as parameter and asking an AI to compress it as it can and then decompress it to get the output cube, which I will call the "Houglass" method.