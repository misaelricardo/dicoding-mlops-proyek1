да
На
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8ЫТ
А
Adam/dense_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_48/bias/v
y
(Adam/dense_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_48/kernel/v
Б
*Adam/dense_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/v*
_output_shapes

: *
dtype0
А
Adam/dense_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_47/bias/v
y
(Adam/dense_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/v*
_output_shapes
: *
dtype0
И
Adam/dense_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_47/kernel/v
Б
*Adam/dense_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/v*
_output_shapes

:@ *
dtype0
А
Adam/dense_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_46/bias/v
y
(Adam/dense_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/v*
_output_shapes
:@*
dtype0
И
Adam/dense_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_46/kernel/v
Б
*Adam/dense_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/v*
_output_shapes

:@*
dtype0
А
Adam/dense_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_48/bias/m
y
(Adam/dense_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_48/kernel/m
Б
*Adam/dense_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/m*
_output_shapes

: *
dtype0
А
Adam/dense_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_47/bias/m
y
(Adam/dense_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/m*
_output_shapes
: *
dtype0
И
Adam/dense_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_47/kernel/m
Б
*Adam/dense_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/m*
_output_shapes

:@ *
dtype0
А
Adam/dense_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_46/bias/m
y
(Adam/dense_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/m*
_output_shapes
:@*
dtype0
И
Adam/dense_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_46/kernel/m
Б
*Adam/dense_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/m*
_output_shapes

:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_48/bias
k
!dense_48/bias/Read/ReadVariableOpReadVariableOpdense_48/bias*
_output_shapes
:*
dtype0
z
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_48/kernel
s
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel*
_output_shapes

: *
dtype0
r
dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_47/bias
k
!dense_47/bias/Read/ReadVariableOpReadVariableOpdense_47/bias*
_output_shapes
: *
dtype0
z
dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_47/kernel
s
#dense_47/kernel/Read/ReadVariableOpReadVariableOpdense_47/kernel*
_output_shapes

:@ *
dtype0
r
dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_46/bias
k
!dense_46/bias/Read/ReadVariableOpReadVariableOpdense_46/bias*
_output_shapes
:@*
dtype0
z
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_46/kernel
s
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel*
_output_shapes

:@*
dtype0
}
serving_default_alcohol_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

serving_default_chlorides_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Б
serving_default_citric_acid_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
}
serving_default_density_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Г
 serving_default_fixed_acidity_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Й
&serving_default_free_sulfur_dioxide_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
x
serving_default_ph_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Д
!serving_default_residual_sugar_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

serving_default_sulphates_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
К
'serving_default_total_sulfur_dioxide_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Ж
#serving_default_volatile_acidity_xfPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
х
StatefulPartitionedCallStatefulPartitionedCallserving_default_alcohol_xfserving_default_chlorides_xfserving_default_citric_acid_xfserving_default_density_xf serving_default_fixed_acidity_xf&serving_default_free_sulfur_dioxide_xfserving_default_ph_xf!serving_default_residual_sugar_xfserving_default_sulphates_xf'serving_default_total_sulfur_dioxide_xf#serving_default_volatile_acidity_xfdense_46/kerneldense_46/biasdense_47/kerneldense_47/biasdense_48/kerneldense_48/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_131823

NoOpNoOp
П5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 4
valueј4Bљ4 Bґ4
в
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¶
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
¶
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
¶
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
.
%0
&1
-2
.3
54
65*
.
%0
&1
-2
.3
54
65*
* 
∞
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
∞
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate%mq&mr-ms.mt5mu6mv%vw&vx-vy.vz5v{6v|*

Iserving_default* 
* 
* 
* 
С
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Otrace_0* 

Ptrace_0* 

%0
&1*

%0
&1*
* 
У
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 
_Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_46/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
У
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
_Y
VARIABLE_VALUEdense_47/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_47/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 
У
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
_Y
VARIABLE_VALUEdense_48/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_48/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

f0
g1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
h	variables
i	keras_api
	jtotal
	kcount*
H
l	variables
m	keras_api
	ntotal
	ocount
p
_fn_kwargs*

j0
k1*

h	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

n0
o1*

l	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
В|
VARIABLE_VALUEAdam/dense_46/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_46/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_47/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_47/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_48/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_48/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_46/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_46/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_47/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_47/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_48/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_48/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ђ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_46/kernel/Read/ReadVariableOp!dense_46/bias/Read/ReadVariableOp#dense_47/kernel/Read/ReadVariableOp!dense_47/bias/Read/ReadVariableOp#dense_48/kernel/Read/ReadVariableOp!dense_48/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_46/kernel/m/Read/ReadVariableOp(Adam/dense_46/bias/m/Read/ReadVariableOp*Adam/dense_47/kernel/m/Read/ReadVariableOp(Adam/dense_47/bias/m/Read/ReadVariableOp*Adam/dense_48/kernel/m/Read/ReadVariableOp(Adam/dense_48/bias/m/Read/ReadVariableOp*Adam/dense_46/kernel/v/Read/ReadVariableOp(Adam/dense_46/bias/v/Read/ReadVariableOp*Adam/dense_47/kernel/v/Read/ReadVariableOp(Adam/dense_47/bias/v/Read/ReadVariableOp*Adam/dense_48/kernel/v/Read/ReadVariableOp(Adam/dense_48/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_132156
К
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_46/kerneldense_46/biasdense_47/kerneldense_47/biasdense_48/kerneldense_48/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_46/kernel/mAdam/dense_46/bias/mAdam/dense_47/kernel/mAdam/dense_47/bias/mAdam/dense_48/kernel/mAdam/dense_48/bias/mAdam/dense_46/kernel/vAdam/dense_46/bias/vAdam/dense_47/kernel/vAdam/dense_47/bias/vAdam/dense_48/kernel/vAdam/dense_48/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_132247вИ
њ 
и
D__inference_model_15_layer_call_and_return_conditional_losses_131788
fixed_acidity_xf
volatile_acidity_xf
citric_acid_xf
residual_sugar_xf
chlorides_xf
free_sulfur_dioxide_xf
total_sulfur_dioxide_xf

density_xf	
ph_xf
sulphates_xf

alcohol_xf!
dense_46_131772:@
dense_46_131774:@!
dense_47_131777:@ 
dense_47_131779: !
dense_48_131782: 
dense_48_131784:
identityИҐ dense_46/StatefulPartitionedCallҐ dense_47/StatefulPartitionedCallҐ dense_48/StatefulPartitionedCallь
concatenate_16/PartitionedCallPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131502С
 dense_46/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0dense_46_131772dense_46_131774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_131515У
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_131777dense_47_131779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_131532У
 dense_48/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0dense_48_131782dense_48_131784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_131549x
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѓ
NoOpNoOp!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameph_xf:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
alcohol_xf
√.
§
!__inference__wrapped_model_131459
fixed_acidity_xf
volatile_acidity_xf
citric_acid_xf
residual_sugar_xf
chlorides_xf
free_sulfur_dioxide_xf
total_sulfur_dioxide_xf

density_xf	
ph_xf
sulphates_xf

alcohol_xfB
0model_15_dense_46_matmul_readvariableop_resource:@?
1model_15_dense_46_biasadd_readvariableop_resource:@B
0model_15_dense_47_matmul_readvariableop_resource:@ ?
1model_15_dense_47_biasadd_readvariableop_resource: B
0model_15_dense_48_matmul_readvariableop_resource: ?
1model_15_dense_48_biasadd_readvariableop_resource:
identityИҐ(model_15/dense_46/BiasAdd/ReadVariableOpҐ'model_15/dense_46/MatMul/ReadVariableOpҐ(model_15/dense_47/BiasAdd/ReadVariableOpҐ'model_15/dense_47/MatMul/ReadVariableOpҐ(model_15/dense_48/BiasAdd/ReadVariableOpҐ'model_15/dense_48/MatMul/ReadVariableOpe
#model_15/concatenate_16/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :…
model_15/concatenate_16/concatConcatV2fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf,model_15/concatenate_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Ш
'model_15/dense_46/MatMul/ReadVariableOpReadVariableOp0model_15_dense_46_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ѓ
model_15/dense_46/MatMulMatMul'model_15/concatenate_16/concat:output:0/model_15/dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(model_15/dense_46/BiasAdd/ReadVariableOpReadVariableOp1model_15_dense_46_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
model_15/dense_46/BiasAddBiasAdd"model_15/dense_46/MatMul:product:00model_15/dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@t
model_15/dense_46/ReluRelu"model_15/dense_46/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
'model_15/dense_47/MatMul/ReadVariableOpReadVariableOp0model_15_dense_47_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ђ
model_15/dense_47/MatMulMatMul$model_15/dense_46/Relu:activations:0/model_15/dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(model_15/dense_47/BiasAdd/ReadVariableOpReadVariableOp1model_15_dense_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
model_15/dense_47/BiasAddBiasAdd"model_15/dense_47/MatMul:product:00model_15/dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ t
model_15/dense_47/ReluRelu"model_15/dense_47/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
'model_15/dense_48/MatMul/ReadVariableOpReadVariableOp0model_15_dense_48_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ђ
model_15/dense_48/MatMulMatMul$model_15/dense_47/Relu:activations:0/model_15/dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(model_15/dense_48/BiasAdd/ReadVariableOpReadVariableOp1model_15_dense_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
model_15/dense_48/BiasAddBiasAdd"model_15/dense_48/MatMul:product:00model_15/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€z
model_15/dense_48/SigmoidSigmoid"model_15/dense_48/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
IdentityIdentitymodel_15/dense_48/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≈
NoOpNoOp)^model_15/dense_46/BiasAdd/ReadVariableOp(^model_15/dense_46/MatMul/ReadVariableOp)^model_15/dense_47/BiasAdd/ReadVariableOp(^model_15/dense_47/MatMul/ReadVariableOp)^model_15/dense_48/BiasAdd/ReadVariableOp(^model_15/dense_48/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2T
(model_15/dense_46/BiasAdd/ReadVariableOp(model_15/dense_46/BiasAdd/ReadVariableOp2R
'model_15/dense_46/MatMul/ReadVariableOp'model_15/dense_46/MatMul/ReadVariableOp2T
(model_15/dense_47/BiasAdd/ReadVariableOp(model_15/dense_47/BiasAdd/ReadVariableOp2R
'model_15/dense_47/MatMul/ReadVariableOp'model_15/dense_47/MatMul/ReadVariableOp2T
(model_15/dense_48/BiasAdd/ReadVariableOp(model_15/dense_48/BiasAdd/ReadVariableOp2R
'model_15/dense_48/MatMul/ReadVariableOp'model_15/dense_48/MatMul/ReadVariableOp:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameph_xf:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
alcohol_xf
Ы

х
D__inference_dense_46_layer_call_and_return_conditional_losses_131515

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
Ў
)__inference_model_15_layer_call_fn_131728
fixed_acidity_xf
volatile_acidity_xf
citric_acid_xf
residual_sugar_xf
chlorides_xf
free_sulfur_dioxide_xf
total_sulfur_dioxide_xf

density_xf	
ph_xf
sulphates_xf

alcohol_xf
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_131686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameph_xf:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
alcohol_xf
Ю
у
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131502

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :–
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Џ;
К
__inference__traced_save_132156
file_prefix.
*savev2_dense_46_kernel_read_readvariableop,
(savev2_dense_46_bias_read_readvariableop.
*savev2_dense_47_kernel_read_readvariableop,
(savev2_dense_47_bias_read_readvariableop.
*savev2_dense_48_kernel_read_readvariableop,
(savev2_dense_48_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_46_kernel_m_read_readvariableop3
/savev2_adam_dense_46_bias_m_read_readvariableop5
1savev2_adam_dense_47_kernel_m_read_readvariableop3
/savev2_adam_dense_47_bias_m_read_readvariableop5
1savev2_adam_dense_48_kernel_m_read_readvariableop3
/savev2_adam_dense_48_bias_m_read_readvariableop5
1savev2_adam_dense_46_kernel_v_read_readvariableop3
/savev2_adam_dense_46_bias_v_read_readvariableop5
1savev2_adam_dense_47_kernel_v_read_readvariableop3
/savev2_adam_dense_47_bias_v_read_readvariableop5
1savev2_adam_dense_48_kernel_v_read_readvariableop3
/savev2_adam_dense_48_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: х
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH•
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B э

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_46_kernel_read_readvariableop(savev2_dense_46_bias_read_readvariableop*savev2_dense_47_kernel_read_readvariableop(savev2_dense_47_bias_read_readvariableop*savev2_dense_48_kernel_read_readvariableop(savev2_dense_48_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_46_kernel_m_read_readvariableop/savev2_adam_dense_46_bias_m_read_readvariableop1savev2_adam_dense_47_kernel_m_read_readvariableop/savev2_adam_dense_47_bias_m_read_readvariableop1savev2_adam_dense_48_kernel_m_read_readvariableop/savev2_adam_dense_48_bias_m_read_readvariableop1savev2_adam_dense_46_kernel_v_read_readvariableop/savev2_adam_dense_46_bias_v_read_readvariableop1savev2_adam_dense_47_kernel_v_read_readvariableop/savev2_adam_dense_47_bias_v_read_readvariableop1savev2_adam_dense_48_kernel_v_read_readvariableop/savev2_adam_dense_48_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ї
_input_shapes©
¶: :@:@:@ : : :: : : : : : : : : :@:@:@ : : ::@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
Ѕ
Ў
)__inference_model_15_layer_call_fn_131571
fixed_acidity_xf
volatile_acidity_xf
citric_acid_xf
residual_sugar_xf
chlorides_xf
free_sulfur_dioxide_xf
total_sulfur_dioxide_xf

density_xf	
ph_xf
sulphates_xf

alcohol_xf
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_131556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameph_xf:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
alcohol_xf
Ы(
Ф
D__inference_model_15_layer_call_and_return_conditional_losses_131914
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_109
'dense_46_matmul_readvariableop_resource:@6
(dense_46_biasadd_readvariableop_resource:@9
'dense_47_matmul_readvariableop_resource:@ 6
(dense_47_biasadd_readvariableop_resource: 9
'dense_48_matmul_readvariableop_resource: 6
(dense_48_biasadd_readvariableop_resource:
identityИҐdense_46/BiasAdd/ReadVariableOpҐdense_46/MatMul/ReadVariableOpҐdense_47/BiasAdd/ReadVariableOpҐdense_47/MatMul/ReadVariableOpҐdense_48/BiasAdd/ReadVariableOpҐdense_48/MatMul/ReadVariableOp\
concatenate_16/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :р
concatenate_16/concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10#concatenate_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0У
dense_46/MatMulMatMulconcatenate_16/concat:output:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@b
dense_46/ReluReludense_46/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Р
dense_47/MatMulMatMuldense_46/Relu:activations:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_47/ReluReludense_47/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Р
dense_48/MatMulMatMuldense_47/Relu:activations:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
dense_48/SigmoidSigmoiddense_48/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
IdentityIdentitydense_48/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€П
NoOpNoOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/10
Ы

х
D__inference_dense_46_layer_call_and_return_conditional_losses_132002

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ

х
D__inference_dense_48_layer_call_and_return_conditional_losses_132042

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
њ 
и
D__inference_model_15_layer_call_and_return_conditional_losses_131758
fixed_acidity_xf
volatile_acidity_xf
citric_acid_xf
residual_sugar_xf
chlorides_xf
free_sulfur_dioxide_xf
total_sulfur_dioxide_xf

density_xf	
ph_xf
sulphates_xf

alcohol_xf!
dense_46_131742:@
dense_46_131744:@!
dense_47_131747:@ 
dense_47_131749: !
dense_48_131752: 
dense_48_131754:
identityИҐ dense_46/StatefulPartitionedCallҐ dense_47/StatefulPartitionedCallҐ dense_48/StatefulPartitionedCallь
concatenate_16/PartitionedCallPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131502С
 dense_46/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0dense_46_131742dense_46_131744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_131515У
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_131747dense_47_131749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_131532У
 dense_48/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0dense_48_131752dense_48_131754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_131549x
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѓ
NoOpNoOp!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameph_xf:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
alcohol_xf
м
С
)__inference_model_15_layer_call_fn_131877
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_131686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/10
ѕ
Я
D__inference_model_15_layer_call_and_return_conditional_losses_131686

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10!
dense_46_131670:@
dense_46_131672:@!
dense_47_131675:@ 
dense_47_131677: !
dense_48_131680: 
dense_48_131682:
identityИҐ dense_46/StatefulPartitionedCallҐ dense_47/StatefulPartitionedCallҐ dense_48/StatefulPartitionedCall≥
concatenate_16/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131502С
 dense_46/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0dense_46_131670dense_46_131672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_131515У
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_131675dense_47_131677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_131532У
 dense_48/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0dense_48_131680dense_48_131682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_131549x
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѓ
NoOpNoOp!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ђ
Џ
/__inference_concatenate_16_layer_call_fn_131966
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
identity¶
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131502`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/10
Ы

х
D__inference_dense_47_layer_call_and_return_conditional_losses_131532

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
¬
Ц
)__inference_dense_47_layer_call_fn_132011

inputs
unknown:@ 
	unknown_0: 
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_131532o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
¬
Ц
)__inference_dense_46_layer_call_fn_131991

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_131515o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¬
Ц
)__inference_dense_48_layer_call_fn_132031

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_131549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ы(
Ф
D__inference_model_15_layer_call_and_return_conditional_losses_131951
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_109
'dense_46_matmul_readvariableop_resource:@6
(dense_46_biasadd_readvariableop_resource:@9
'dense_47_matmul_readvariableop_resource:@ 6
(dense_47_biasadd_readvariableop_resource: 9
'dense_48_matmul_readvariableop_resource: 6
(dense_48_biasadd_readvariableop_resource:
identityИҐdense_46/BiasAdd/ReadVariableOpҐdense_46/MatMul/ReadVariableOpҐdense_47/BiasAdd/ReadVariableOpҐdense_47/MatMul/ReadVariableOpҐdense_48/BiasAdd/ReadVariableOpҐdense_48/MatMul/ReadVariableOp\
concatenate_16/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :р
concatenate_16/concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10#concatenate_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0У
dense_46/MatMulMatMulconcatenate_16/concat:output:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@b
dense_46/ReluReludense_46/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Р
dense_47/MatMulMatMuldense_46/Relu:activations:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_47/ReluReludense_47/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Р
dense_48/MatMulMatMuldense_47/Relu:activations:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
dense_48/SigmoidSigmoiddense_48/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
IdentityIdentitydense_48/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€П
NoOpNoOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/10
є
х
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131982
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :“
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/10
…l
т
"__inference__traced_restore_132247
file_prefix2
 assignvariableop_dense_46_kernel:@.
 assignvariableop_1_dense_46_bias:@4
"assignvariableop_2_dense_47_kernel:@ .
 assignvariableop_3_dense_47_bias: 4
"assignvariableop_4_dense_48_kernel: .
 assignvariableop_5_dense_48_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_46_kernel_m:@6
(assignvariableop_16_adam_dense_46_bias_m:@<
*assignvariableop_17_adam_dense_47_kernel_m:@ 6
(assignvariableop_18_adam_dense_47_bias_m: <
*assignvariableop_19_adam_dense_48_kernel_m: 6
(assignvariableop_20_adam_dense_48_bias_m:<
*assignvariableop_21_adam_dense_46_kernel_v:@6
(assignvariableop_22_adam_dense_46_bias_v:@<
*assignvariableop_23_adam_dense_47_kernel_v:@ 6
(assignvariableop_24_adam_dense_47_bias_v: <
*assignvariableop_25_adam_dense_48_kernel_v: 6
(assignvariableop_26_adam_dense_48_bias_v:
identity_28ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ш
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH®
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ђ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_dense_46_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_46_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_47_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_47_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_48_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_48_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_46_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_46_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_47_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_47_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_48_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_48_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_46_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_46_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_47_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_47_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_48_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_48_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 °
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: О
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ы

х
D__inference_dense_47_layer_call_and_return_conditional_losses_132022

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ъ

х
D__inference_dense_48_layer_call_and_return_conditional_losses_131549

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Щ
”
$__inference_signature_wrapper_131823

alcohol_xf
chlorides_xf
citric_acid_xf

density_xf
fixed_acidity_xf
free_sulfur_dioxide_xf	
ph_xf
residual_sugar_xf
sulphates_xf
total_sulfur_dioxide_xf
volatile_acidity_xf
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_131459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
alcohol_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namechlorides_xf:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namecitric_acid_xf:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
density_xf:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefixed_acidity_xf:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namefree_sulfur_dioxide_xf:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameph_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namesulphates_xf:`	\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nametotal_sulfur_dioxide_xf:\
X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namevolatile_acidity_xf
ѕ
Я
D__inference_model_15_layer_call_and_return_conditional_losses_131556

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10!
dense_46_131516:@
dense_46_131518:@!
dense_47_131533:@ 
dense_47_131535: !
dense_48_131550: 
dense_48_131552:
identityИҐ dense_46/StatefulPartitionedCallҐ dense_47/StatefulPartitionedCallҐ dense_48/StatefulPartitionedCall≥
concatenate_16/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131502С
 dense_46/StatefulPartitionedCallStatefulPartitionedCall'concatenate_16/PartitionedCall:output:0dense_46_131516dense_46_131518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_131515У
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_131533dense_47_131535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_131532У
 dense_48/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0dense_48_131550dense_48_131552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_131549x
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѓ
NoOpNoOp!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м
С
)__inference_model_15_layer_call_fn_131850
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_131556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesа
Ё:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/10"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≥
serving_defaultЯ
A

alcohol_xf3
serving_default_alcohol_xf:0€€€€€€€€€
E
chlorides_xf5
serving_default_chlorides_xf:0€€€€€€€€€
I
citric_acid_xf7
 serving_default_citric_acid_xf:0€€€€€€€€€
A

density_xf3
serving_default_density_xf:0€€€€€€€€€
M
fixed_acidity_xf9
"serving_default_fixed_acidity_xf:0€€€€€€€€€
Y
free_sulfur_dioxide_xf?
(serving_default_free_sulfur_dioxide_xf:0€€€€€€€€€
7
ph_xf.
serving_default_ph_xf:0€€€€€€€€€
O
residual_sugar_xf:
#serving_default_residual_sugar_xf:0€€€€€€€€€
E
sulphates_xf5
serving_default_sulphates_xf:0€€€€€€€€€
[
total_sulfur_dioxide_xf@
)serving_default_total_sulfur_dioxide_xf:0€€€€€€€€€
S
volatile_acidity_xf<
%serving_default_volatile_acidity_xf:0€€€€€€€€€<
dense_480
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ћґ
щ
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
•
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
ї
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
ї
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
J
%0
&1
-2
.3
54
65"
trackable_list_wrapper
J
%0
&1
-2
.3
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
 
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ў
<trace_0
=trace_1
>trace_2
?trace_32о
)__inference_model_15_layer_call_fn_131571
)__inference_model_15_layer_call_fn_131850
)__inference_model_15_layer_call_fn_131877
)__inference_model_15_layer_call_fn_131728њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z<trace_0z=trace_1z>trace_2z?trace_3
≈
@trace_0
Atrace_1
Btrace_2
Ctrace_32Џ
D__inference_model_15_layer_call_and_return_conditional_losses_131914
D__inference_model_15_layer_call_and_return_conditional_losses_131951
D__inference_model_15_layer_call_and_return_conditional_losses_131758
D__inference_model_15_layer_call_and_return_conditional_losses_131788њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
щBц
!__inference__wrapped_model_131459fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
њ
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate%mq&mr-ms.mt5mu6mv%vw&vx-vy.vz5v{6v|"
	optimizer
,
Iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
у
Otrace_02÷
/__inference_concatenate_16_layer_call_fn_131966Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zOtrace_0
О
Ptrace_02с
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131982Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zPtrace_0
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
н
Vtrace_02–
)__inference_dense_46_layer_call_fn_131991Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zVtrace_0
И
Wtrace_02л
D__inference_dense_46_layer_call_and_return_conditional_losses_132002Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zWtrace_0
!:@2dense_46/kernel
:@2dense_46/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
н
]trace_02–
)__inference_dense_47_layer_call_fn_132011Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z]trace_0
И
^trace_02л
D__inference_dense_47_layer_call_and_return_conditional_losses_132022Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z^trace_0
!:@ 2dense_47/kernel
: 2dense_47/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
н
dtrace_02–
)__inference_dense_48_layer_call_fn_132031Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zdtrace_0
И
etrace_02л
D__inference_dense_48_layer_call_and_return_conditional_losses_132042Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zetrace_0
!: 2dense_48/kernel
:2dense_48/bias
 "
trackable_list_wrapper
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
®B•
)__inference_model_15_layer_call_fn_131571fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
бBё
)__inference_model_15_layer_call_fn_131850inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
бBё
)__inference_model_15_layer_call_fn_131877inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®B•
)__inference_model_15_layer_call_fn_131728fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
D__inference_model_15_layer_call_and_return_conditional_losses_131914inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
D__inference_model_15_layer_call_and_return_conditional_losses_131951inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
√Bј
D__inference_model_15_layer_call_and_return_conditional_losses_131758fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
√Bј
D__inference_model_15_layer_call_and_return_conditional_losses_131788fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
цBу
$__inference_signature_wrapper_131823
alcohol_xfchlorides_xfcitric_acid_xf
density_xffixed_acidity_xffree_sulfur_dioxide_xfph_xfresidual_sugar_xfsulphates_xftotal_sulfur_dioxide_xfvolatile_acidity_xf"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 B«
/__inference_concatenate_16_layer_call_fn_131966inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131982inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_46_layer_call_fn_131991inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_46_layer_call_and_return_conditional_losses_132002inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_47_layer_call_fn_132011inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_47_layer_call_and_return_conditional_losses_132022inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_48_layer_call_fn_132031inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_48_layer_call_and_return_conditional_losses_132042inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
N
h	variables
i	keras_api
	jtotal
	kcount"
_tf_keras_metric
^
l	variables
m	keras_api
	ntotal
	ocount
p
_fn_kwargs"
_tf_keras_metric
.
j0
k1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$@2Adam/dense_46/kernel/m
 :@2Adam/dense_46/bias/m
&:$@ 2Adam/dense_47/kernel/m
 : 2Adam/dense_47/bias/m
&:$ 2Adam/dense_48/kernel/m
 :2Adam/dense_48/bias/m
&:$@2Adam/dense_46/kernel/v
 :@2Adam/dense_46/bias/v
&:$@ 2Adam/dense_47/kernel/v
 : 2Adam/dense_47/bias/v
&:$ 2Adam/dense_48/kernel/v
 :2Adam/dense_48/bias/v—
!__inference__wrapped_model_131459Ђ%&-.56лҐз
яҐџ
ЎЪ‘
*К'
fixed_acidity_xf€€€€€€€€€
-К*
volatile_acidity_xf€€€€€€€€€
(К%
citric_acid_xf€€€€€€€€€
+К(
residual_sugar_xf€€€€€€€€€
&К#
chlorides_xf€€€€€€€€€
0К-
free_sulfur_dioxide_xf€€€€€€€€€
1К.
total_sulfur_dioxide_xf€€€€€€€€€
$К!

density_xf€€€€€€€€€
К
ph_xf€€€€€€€€€
&К#
sulphates_xf€€€€€€€€€
$К!

alcohol_xf€€€€€€€€€
™ "3™0
.
dense_48"К
dense_48€€€€€€€€€Э
J__inference_concatenate_16_layer_call_and_return_conditional_losses_131982ќ§Ґ†
ШҐФ
СЪН
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
"К
inputs/5€€€€€€€€€
"К
inputs/6€€€€€€€€€
"К
inputs/7€€€€€€€€€
"К
inputs/8€€€€€€€€€
"К
inputs/9€€€€€€€€€
#К 
	inputs/10€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ х
/__inference_concatenate_16_layer_call_fn_131966Ѕ§Ґ†
ШҐФ
СЪН
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
"К
inputs/5€€€€€€€€€
"К
inputs/6€€€€€€€€€
"К
inputs/7€€€€€€€€€
"К
inputs/8€€€€€€€€€
"К
inputs/9€€€€€€€€€
#К 
	inputs/10€€€€€€€€€
™ "К€€€€€€€€€§
D__inference_dense_46_layer_call_and_return_conditional_losses_132002\%&/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
)__inference_dense_46_layer_call_fn_131991O%&/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€@§
D__inference_dense_47_layer_call_and_return_conditional_losses_132022\-./Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ |
)__inference_dense_47_layer_call_fn_132011O-./Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ §
D__inference_dense_48_layer_call_and_return_conditional_losses_132042\56/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_48_layer_call_fn_132031O56/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€о
D__inference_model_15_layer_call_and_return_conditional_losses_131758•%&-.56уҐп
зҐг
ЎЪ‘
*К'
fixed_acidity_xf€€€€€€€€€
-К*
volatile_acidity_xf€€€€€€€€€
(К%
citric_acid_xf€€€€€€€€€
+К(
residual_sugar_xf€€€€€€€€€
&К#
chlorides_xf€€€€€€€€€
0К-
free_sulfur_dioxide_xf€€€€€€€€€
1К.
total_sulfur_dioxide_xf€€€€€€€€€
$К!

density_xf€€€€€€€€€
К
ph_xf€€€€€€€€€
&К#
sulphates_xf€€€€€€€€€
$К!

alcohol_xf€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ о
D__inference_model_15_layer_call_and_return_conditional_losses_131788•%&-.56уҐп
зҐг
ЎЪ‘
*К'
fixed_acidity_xf€€€€€€€€€
-К*
volatile_acidity_xf€€€€€€€€€
(К%
citric_acid_xf€€€€€€€€€
+К(
residual_sugar_xf€€€€€€€€€
&К#
chlorides_xf€€€€€€€€€
0К-
free_sulfur_dioxide_xf€€€€€€€€€
1К.
total_sulfur_dioxide_xf€€€€€€€€€
$К!

density_xf€€€€€€€€€
К
ph_xf€€€€€€€€€
&К#
sulphates_xf€€€€€€€€€
$К!

alcohol_xf€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ І
D__inference_model_15_layer_call_and_return_conditional_losses_131914ё%&-.56ђҐ®
†ҐЬ
СЪН
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
"К
inputs/5€€€€€€€€€
"К
inputs/6€€€€€€€€€
"К
inputs/7€€€€€€€€€
"К
inputs/8€€€€€€€€€
"К
inputs/9€€€€€€€€€
#К 
	inputs/10€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ І
D__inference_model_15_layer_call_and_return_conditional_losses_131951ё%&-.56ђҐ®
†ҐЬ
СЪН
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
"К
inputs/5€€€€€€€€€
"К
inputs/6€€€€€€€€€
"К
inputs/7€€€€€€€€€
"К
inputs/8€€€€€€€€€
"К
inputs/9€€€€€€€€€
#К 
	inputs/10€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ∆
)__inference_model_15_layer_call_fn_131571Ш%&-.56уҐп
зҐг
ЎЪ‘
*К'
fixed_acidity_xf€€€€€€€€€
-К*
volatile_acidity_xf€€€€€€€€€
(К%
citric_acid_xf€€€€€€€€€
+К(
residual_sugar_xf€€€€€€€€€
&К#
chlorides_xf€€€€€€€€€
0К-
free_sulfur_dioxide_xf€€€€€€€€€
1К.
total_sulfur_dioxide_xf€€€€€€€€€
$К!

density_xf€€€€€€€€€
К
ph_xf€€€€€€€€€
&К#
sulphates_xf€€€€€€€€€
$К!

alcohol_xf€€€€€€€€€
p 

 
™ "К€€€€€€€€€∆
)__inference_model_15_layer_call_fn_131728Ш%&-.56уҐп
зҐг
ЎЪ‘
*К'
fixed_acidity_xf€€€€€€€€€
-К*
volatile_acidity_xf€€€€€€€€€
(К%
citric_acid_xf€€€€€€€€€
+К(
residual_sugar_xf€€€€€€€€€
&К#
chlorides_xf€€€€€€€€€
0К-
free_sulfur_dioxide_xf€€€€€€€€€
1К.
total_sulfur_dioxide_xf€€€€€€€€€
$К!

density_xf€€€€€€€€€
К
ph_xf€€€€€€€€€
&К#
sulphates_xf€€€€€€€€€
$К!

alcohol_xf€€€€€€€€€
p

 
™ "К€€€€€€€€€€
)__inference_model_15_layer_call_fn_131850—%&-.56ђҐ®
†ҐЬ
СЪН
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
"К
inputs/5€€€€€€€€€
"К
inputs/6€€€€€€€€€
"К
inputs/7€€€€€€€€€
"К
inputs/8€€€€€€€€€
"К
inputs/9€€€€€€€€€
#К 
	inputs/10€€€€€€€€€
p 

 
™ "К€€€€€€€€€€
)__inference_model_15_layer_call_fn_131877—%&-.56ђҐ®
†ҐЬ
СЪН
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
"К
inputs/5€€€€€€€€€
"К
inputs/6€€€€€€€€€
"К
inputs/7€€€€€€€€€
"К
inputs/8€€€€€€€€€
"К
inputs/9€€€€€€€€€
#К 
	inputs/10€€€€€€€€€
p

 
™ "К€€€€€€€€€Щ
$__inference_signature_wrapper_131823р%&-.56∞Ґђ
Ґ 
§™†
2

alcohol_xf$К!

alcohol_xf€€€€€€€€€
6
chlorides_xf&К#
chlorides_xf€€€€€€€€€
:
citric_acid_xf(К%
citric_acid_xf€€€€€€€€€
2

density_xf$К!

density_xf€€€€€€€€€
>
fixed_acidity_xf*К'
fixed_acidity_xf€€€€€€€€€
J
free_sulfur_dioxide_xf0К-
free_sulfur_dioxide_xf€€€€€€€€€
(
ph_xfК
ph_xf€€€€€€€€€
@
residual_sugar_xf+К(
residual_sugar_xf€€€€€€€€€
6
sulphates_xf&К#
sulphates_xf€€€€€€€€€
L
total_sulfur_dioxide_xf1К.
total_sulfur_dioxide_xf€€€€€€€€€
D
volatile_acidity_xf-К*
volatile_acidity_xf€€€€€€€€€"3™0
.
dense_48"К
dense_48€€€€€€€€€