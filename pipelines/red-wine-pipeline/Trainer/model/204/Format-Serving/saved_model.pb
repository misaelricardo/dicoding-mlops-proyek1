чґ
Ят
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Р
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
≥
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8ца
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
Ж
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:@*
dtype0
Ъ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_5/beta/v
У
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_5/gamma/v
Х
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:@*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:@@*
dtype0
Ъ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/v
У
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/v
Х
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:@*
dtype0
З
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*&
shared_nameAdam/dense_5/kernel/v
А
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	А@*
dtype0
Ы
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/v
Ф
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/v
Ц
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:А*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_4/kernel/v
А
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	А*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
Ж
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:@*
dtype0
Ъ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_5/beta/m
У
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_5/gamma/m
Х
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:@*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:@@*
dtype0
Ъ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/m
У
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/m
Х
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:@*
dtype0
З
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*&
shared_nameAdam/dense_5/kernel/m
А
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	А@*
dtype0
Ы
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/m
Ф
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/m
Ц
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:А*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_4/kernel/m
А
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	А*
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
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@@*
dtype0
Ґ
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	А@*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:А*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	А*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
№
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesdense_4/kerneldense_4/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_5/kerneldense_5/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_6/kerneldense_6/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_7/kerneldense_7/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_497163

NoOpNoOp
с|
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ђ|
valueҐ|BЯ| BШ|
ї
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
layer-14
layer_with_weights-2
layer-15
layer_with_weights-3
layer-16
layer-17
layer_with_weights-4
layer-18
layer_with_weights-5
layer-19
layer_with_weights-6
layer-20
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

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
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
¶
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
’
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance*
•
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator* 
¶
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
’
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance*
•
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator* 
¶
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
’
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance*
¶
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias*
і
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
${ _saved_model_loader_tracked_dict* 
Ъ
,0
-1
52
63
74
85
F6
G7
O8
P9
Q10
R11
`12
a13
i14
j15
k16
l17
s18
t19*
j
,0
-1
52
63
F4
G5
O6
P7
`8
a9
i10
j11
s12
t13*
* 
±
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Бtrace_0
Вtrace_1
Гtrace_2
Дtrace_3* 
:
Еtrace_0
Жtrace_1
Зtrace_2
Иtrace_3* 
* 
б
	Йiter
Кbeta_1
Лbeta_2

Мdecay
Нlearning_rate,mА-mБ5mВ6mГFmДGmЕOmЖPmЗ`mИamЙimКjmЛsmМtmН,vО-vП5vР6vСFvТGvУOvФPvХ`vЦavЧivШjvЩsvЪtvЫ*

Оserving_default* 
* 
* 
* 
Ц
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

Фtrace_0* 

Хtrace_0* 

,0
-1*

,0
-1*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
50
61
72
83*

50
61*
* 
Ш
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

Ґtrace_0
£trace_1* 

§trace_0
•trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

Ђtrace_0
ђtrace_1* 

≠trace_0
Ѓtrace_1* 
* 

F0
G1*

F0
G1*
* 
Ш
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

іtrace_0* 

µtrace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
O0
P1
Q2
R3*

O0
P1*
* 
Ш
ґnon_trainable_variables
Јlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

їtrace_0
Љtrace_1* 

љtrace_0
Њtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

ƒtrace_0
≈trace_1* 

∆trace_0
«trace_1* 
* 

`0
a1*

`0
a1*
* 
Ш
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Ќtrace_0* 

ќtrace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
i0
j1
k2
l3*

i0
j1*
* 
Ш
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

‘trace_0
’trace_1* 

÷trace_0
„trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

s0
t1*

s0
t1*
* 
Ш
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

дtrace_0
еtrace_1* 

жtrace_0
зtrace_1* 
y
и	_imported
й_wrapped_function
к_structured_inputs
л_structured_outputs
м_output_to_inputs_map* 
.
70
81
Q2
R3
k4
l5*
™
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
14
15
16
17
18
19
20
21*

н0
о1*
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

70
81*
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

Q0
R1*
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

k0
l1*
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
ђ
пcreated_variables
р	resources
сtrackable_objects
тinitializers
уassets
ф
signatures
$х_self_saveable_object_factories
йtransform_fn* 
* 
* 
* 
* 
<
ц	variables
ч	keras_api

шtotal

щcount*
M
ъ	variables
ы	keras_api

ьtotal

эcount
ю
_fn_kwargs*
* 
* 
* 
* 
* 

€serving_default* 
* 

ш0
щ1*

ц	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ь0
э1*

ъ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
Б{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
__inference__traced_save_499375
њ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_5/kerneldense_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_6/kerneldense_6/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_4/kernel/mAdam/dense_4/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/dense_5/kernel/mAdam/dense_5/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/dense_6/kernel/mAdam/dense_6/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/dense_5/kernel/vAdam/dense_5/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/dense_6/kernel/vAdam/dense_6/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*E
Tin>
<2:*
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
"__inference__traced_restore_499556‘Ѕ
у	
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_497914

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ќB
ѓ
C__inference_model_1_layer_call_and_return_conditional_losses_498333
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
dense_4_498283:	А
dense_4_498285:	А+
batch_normalization_3_498288:	А+
batch_normalization_3_498290:	А+
batch_normalization_3_498292:	А+
batch_normalization_3_498294:	А!
dense_5_498298:	А@
dense_5_498300:@*
batch_normalization_4_498303:@*
batch_normalization_4_498305:@*
batch_normalization_4_498307:@*
batch_normalization_4_498309:@ 
dense_6_498313:@@
dense_6_498315:@*
batch_normalization_5_498318:@*
batch_normalization_5_498320:@*
batch_normalization_5_498322:@*
batch_normalization_5_498324:@ 
dense_7_498327:@
dense_7_498329:
identityИҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallъ
concatenate_1/PartitionedCallPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
GPU 2J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_497720Н
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_498283dense_4_498285*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_497733Е
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_498288batch_normalization_3_498290batch_normalization_3_498292batch_normalization_3_498294*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497328ы
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_497947Р
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_5_498298dense_5_498300*
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
GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_497766Д
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_498303batch_normalization_4_498305batch_normalization_4_498307batch_normalization_4_498309*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497410Ю
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_497914Р
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_6_498313dense_6_498315*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_497799Д
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_498318batch_normalization_5_498320batch_normalization_5_498322batch_normalization_5_498324*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497492Ь
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_498327dense_7_498329*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_497824w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Y U
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
т≤
І
'__inference_serve_tf_examples_fn_497116
examplesA
.model_1_dense_4_matmul_readvariableop_resource:	А>
/model_1_dense_4_biasadd_readvariableop_resource:	АN
?model_1_batch_normalization_3_batchnorm_readvariableop_resource:	АR
Cmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resource:	АP
Amodel_1_batch_normalization_3_batchnorm_readvariableop_1_resource:	АP
Amodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource:	АA
.model_1_dense_5_matmul_readvariableop_resource:	А@=
/model_1_dense_5_biasadd_readvariableop_resource:@M
?model_1_batch_normalization_4_batchnorm_readvariableop_resource:@Q
Cmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource:@O
Amodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource:@O
Amodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource:@@
.model_1_dense_6_matmul_readvariableop_resource:@@=
/model_1_dense_6_biasadd_readvariableop_resource:@M
?model_1_batch_normalization_5_batchnorm_readvariableop_resource:@Q
Cmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource:@O
Amodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource:@O
Amodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource:@@
.model_1_dense_7_matmul_readvariableop_resource:@=
/model_1_dense_7_biasadd_readvariableop_resource:
identityИҐ6model_1/batch_normalization_3/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ6model_1/batch_normalization_4/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOpҐ6model_1/batch_normalization_5/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOpҐ&model_1/dense_4/BiasAdd/ReadVariableOpҐ%model_1/dense_4/MatMul/ReadVariableOpҐ&model_1/dense_5/BiasAdd/ReadVariableOpҐ%model_1/dense_5/MatMul/ReadVariableOpҐ&model_1/dense_6/BiasAdd/ReadVariableOpҐ%model_1/dense_6/MatMul/ReadVariableOpҐ&model_1/dense_7/BiasAdd/ReadVariableOpҐ%model_1/dense_7/MatMul/ReadVariableOpU
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_5Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_6Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_7Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_8Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_9Const*
_output_shapes
: *
dtype0*
valueB X
ParseExample/Const_10Const*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB Е
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*™
value†BЭBalcoholB	chloridesBcitric acidBdensityBfixed acidityBfree sulfur dioxideBpHBresidual sugarB	sulphatesBtotal sulfur dioxideBvolatile acidityj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB Б
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0ParseExample/Const_10:output:0*
Tdense
2*з
_output_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*T
dense_shapesD
B:::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 x
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ј
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ј
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€∆
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€€
(transform_features_layer/PartitionedCallPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:2*ParseExample/ParseExampleV2:dense_values:3*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:68transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9+ParseExample/ParseExampleV2:dense_values:10*
Tin
2	*
Tout
2	*ъ
_output_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_pruned_496913c
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
model_1/concatenate_1/concatConcatV21transform_features_layer/PartitionedCall:output:42transform_features_layer/PartitionedCall:output:111transform_features_layer/PartitionedCall:output:21transform_features_layer/PartitionedCall:output:81transform_features_layer/PartitionedCall:output:11transform_features_layer/PartitionedCall:output:52transform_features_layer/PartitionedCall:output:101transform_features_layer/PartitionedCall:output:31transform_features_layer/PartitionedCall:output:61transform_features_layer/PartitionedCall:output:91transform_features_layer/PartitionedCall:output:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Х
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0©
model_1/dense_4/MatMulMatMul%model_1/concatenate_1/concat:output:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АУ
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0І
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аq
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А≥
6model_1/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0r
-model_1/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:“
+model_1/batch_normalization_3/batchnorm/addAddV2>model_1/batch_normalization_3/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:АН
-model_1/batch_normalization_3/batchnorm/RsqrtRsqrt/model_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:Аї
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0ѕ
+model_1/batch_normalization_3/batchnorm/mulMul1model_1/batch_normalization_3/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
-model_1/batch_normalization_3/batchnorm/mul_1Mul"model_1/dense_4/Relu:activations:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€АЈ
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
-model_1/batch_normalization_3/batchnorm/mul_2Mul@model_1/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:АЈ
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0Ќ
+model_1/batch_normalization_3/batchnorm/subSub@model_1/batch_normalization_3/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:АЌ
-model_1/batch_normalization_3/batchnorm/add_1AddV21model_1/batch_normalization_3/batchnorm/mul_1:z:0/model_1/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€АМ
model_1/dropout_2/IdentityIdentity1model_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€АХ
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0¶
model_1/dense_5/MatMulMatMul#model_1/dropout_2/Identity:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Т
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@p
model_1/dense_5/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
6model_1/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_1/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:—
+model_1/batch_normalization_4/batchnorm/addAddV2>model_1/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:@М
-model_1/batch_normalization_4/batchnorm/RsqrtRsqrt/model_1/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:@Ї
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
+model_1/batch_normalization_4/batchnorm/mulMul1model_1/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@ї
-model_1/batch_normalization_4/batchnorm/mul_1Mul"model_1/dense_5/Relu:activations:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
-model_1/batch_normalization_4/batchnorm/mul_2Mul@model_1/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:@ґ
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0ћ
+model_1/batch_normalization_4/batchnorm/subSub@model_1/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@ћ
-model_1/batch_normalization_4/batchnorm/add_1AddV21model_1/batch_normalization_4/batchnorm/mul_1:z:0/model_1/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
model_1/dropout_3/IdentityIdentity1model_1/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0¶
model_1/dense_6/MatMulMatMul#model_1/dropout_3/Identity:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Т
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@p
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
6model_1/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_1/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:—
+model_1/batch_normalization_5/batchnorm/addAddV2>model_1/batch_normalization_5/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:@М
-model_1/batch_normalization_5/batchnorm/RsqrtRsqrt/model_1/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:@Ї
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
+model_1/batch_normalization_5/batchnorm/mulMul1model_1/batch_normalization_5/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@ї
-model_1/batch_normalization_5/batchnorm/mul_1Mul"model_1/dense_6/Relu:activations:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
-model_1/batch_normalization_5/batchnorm/mul_2Mul@model_1/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:@ґ
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0ћ
+model_1/batch_normalization_5/batchnorm/subSub@model_1/batch_normalization_5/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@ћ
-model_1/batch_normalization_5/batchnorm/add_1AddV21model_1/batch_normalization_5/batchnorm/mul_1:z:0/model_1/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0і
model_1/dense_7/MatMulMatMul1model_1/batch_normalization_5/batchnorm/add_1:z:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€o
IdentityIdentity model_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ќ
NoOpNoOp7^model_1/batch_normalization_3/batchnorm/ReadVariableOp9^model_1/batch_normalization_3/batchnorm/ReadVariableOp_19^model_1/batch_normalization_3/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_4/batchnorm/ReadVariableOp9^model_1/batch_normalization_4/batchnorm/ReadVariableOp_19^model_1/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_5/batchnorm/ReadVariableOp9^model_1/batch_normalization_5/batchnorm/ReadVariableOp_19^model_1/batch_normalization_5/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2p
6model_1/batch_normalization_3/batchnorm/ReadVariableOp6model_1/batch_normalization_3/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_18model_1/batch_normalization_3/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_28model_1/batch_normalization_3/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_4/batchnorm/ReadVariableOp6model_1/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_18model_1/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_28model_1/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_5/batchnorm/ReadVariableOp6model_1/batch_normalization_5/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_18model_1/batch_normalization_5/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_28model_1/batch_normalization_5/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp:M I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
examples
ќ
∞
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498915

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Э
т
I__inference_concatenate_1_layer_call_and_return_conditional_losses_497720

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
¶
—
6__inference_batch_normalization_5_layer_call_fn_499022

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497492o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
∞
Г
__inference_pruned_496913

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7	
inputs_8
inputs_9
	inputs_10
	inputs_11
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7	

identity_8

identity_9
identity_10
identity_11Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€\
IdentityIdentityinputs_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_1Identityinputs_1_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_2_copyIdentityinputs_2*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_2Identityinputs_2_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_3_copyIdentityinputs_3*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_3Identityinputs_3_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_4_copyIdentityinputs_4*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_4Identityinputs_4_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_5_copyIdentityinputs_5*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_5Identityinputs_5_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_6_copyIdentityinputs_6*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_6Identityinputs_6_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_7_copyIdentityinputs_7*
T0	*'
_output_shapes
:€€€€€€€€€`

Identity_7Identityinputs_7_copy:output:0*
T0	*'
_output_shapes
:€€€€€€€€€U
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_8Identityinputs_8_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_9_copyIdentityinputs_9*
T0*'
_output_shapes
:€€€€€€€€€`

Identity_9Identityinputs_9_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:€€€€€€€€€b
Identity_10Identityinputs_10_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€W
inputs_11_copyIdentity	inputs_11*
T0*'
_output_shapes
:€€€€€€€€€b
Identity_11Identityinputs_11_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*щ
_input_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:- )
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-	)
'
_output_shapes
:€€€€€€€€€:-
)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€
Ъ

ф
C__inference_dense_6_layer_call_and_return_conditional_losses_498996

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
а
і
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497281

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€АЇ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ў
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_497786

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ѓ%
о
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498822

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	АИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ађ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Аі
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ак
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
У
с
$__inference_signature_wrapper_497163
examples
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *0
f+R)
'__inference_serve_tf_examples_fn_497116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
examples
ќ
∞
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499042

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ў
љ
9__inference_transform_features_layer_layer_call_fn_499130
inputs_alcohol
inputs_chlorides
placeholder
inputs_density
placeholder_1
placeholder_2
	inputs_ph
placeholder_3
inputs_sulphates
placeholder_4
placeholder_5
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10≥
PartitionedCallPartitionedCallinputs_alcoholinputs_chloridesplaceholderinputs_densityplaceholder_1placeholder_2	inputs_phplaceholder_3inputs_sulphatesplaceholder_4placeholder_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *з
_output_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497568`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_7IdentityPartitionedCall:output:7*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs/alcohol:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinputs/chlorides:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs/citric acid:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs/density:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs/fixed acidity:c_
'
_output_shapes
:€€€€€€€€€
4
_user_specified_nameinputs/free sulfur dioxide:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/pH:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameinputs/residual sugar:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinputs/sulphates:d	`
'
_output_shapes
:€€€€€€€€€
5
_user_specified_nameinputs/total sulfur dioxide:`
\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs/volatile acidity
х
c
*__inference_dropout_2_layer_call_fn_498832

inputs
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_497947p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Т%
к
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497410

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ј
Х
(__inference_dense_6_layer_call_fn_498985

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallЎ
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_497799o
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
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Я
F
*__inference_dropout_3_layer_call_fn_498954

inputs
identity∞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_497786`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ї÷
¬
C__inference_model_1_layer_call_and_return_conditional_losses_498691
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
&dense_4_matmul_readvariableop_resource:	А6
'dense_4_biasadd_readvariableop_resource:	АL
=batch_normalization_3_assignmovingavg_readvariableop_resource:	АN
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:	АJ
;batch_normalization_3_batchnorm_mul_readvariableop_resource:	АF
7batch_normalization_3_batchnorm_readvariableop_resource:	А9
&dense_5_matmul_readvariableop_resource:	А@5
'dense_5_biasadd_readvariableop_resource:@K
=batch_normalization_4_assignmovingavg_readvariableop_resource:@M
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_4_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_4_batchnorm_readvariableop_resource:@8
&dense_6_matmul_readvariableop_resource:@@5
'dense_6_biasadd_readvariableop_resource:@K
=batch_normalization_5_assignmovingavg_readvariableop_resource:@M
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_5_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_5_batchnorm_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identityИҐ%batch_normalization_3/AssignMovingAvgҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_3/AssignMovingAvg_1Ґ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐ%batch_normalization_4/AssignMovingAvgҐ4batch_normalization_4/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_4/AssignMovingAvg_1Ґ6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐ%batch_normalization_5/AssignMovingAvgҐ4batch_normalization_5/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_5/AssignMovingAvg_1Ґ6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_5/batchnorm/ReadVariableOpҐ2batch_normalization_5/batchnorm/mul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOp[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
concatenate_1/concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0С
dense_4/MatMulMatMulconcatenate_1/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ј
"batch_normalization_3/moments/meanMeandense_4/Relu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(С
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	А»
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_4/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€АВ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: б
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(Ъ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 †
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0ƒ
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes	
:Аї
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:АД
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≥
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0 
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:АЅ
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:АМ
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:і
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А}
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:АЂ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ј
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А§
%batch_normalization_3/batchnorm/mul_1Muldense_4/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€АЂ
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:А£
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0≥
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аµ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ь
dropout_2/dropout/MulMul)batch_normalization_3/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
dropout_2/dropout/ShapeShape)batch_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:°
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>≈
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АД
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€АИ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0О
dense_5/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: њ
"batch_normalization_4/moments/meanMeandense_5/Relu:activations:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(Р
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:@«
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_5/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: а
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(Щ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Я
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ѓ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0√
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:@Ї
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Д
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0…
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@ј
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@М
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:≥
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:@™
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@£
%batch_normalization_4/batchnorm/mul_1Muldense_5/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@™
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:@Ґ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0≤
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@і
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ы
dropout_3/dropout/MulMul)batch_normalization_4/batchnorm/add_1:z:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@p
dropout_3/dropout/ShapeShape)batch_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:†
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>ƒ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@З
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0О
dense_6/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: њ
"batch_normalization_5/moments/meanMeandense_6/Relu:activations:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(Р
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:@«
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: а
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(Щ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Я
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ѓ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0√
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes
:@Ї
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Д
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≤
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0…
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@ј
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@М
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:≥
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:@™
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@£
%batch_normalization_5/batchnorm/mul_1Muldense_6/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@™
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:@Ґ
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0≤
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@і
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ь
dense_7/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€¬	
NoOpNoOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Q M
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
™
ў
.__inference_concatenate_1_layer_call_fn_498706
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
identity•
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
GPU 2J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_497720`
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
®
—
6__inference_batch_normalization_5_layer_call_fn_499009

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
№
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_498837

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ

ц
C__inference_dense_4_layer_call_and_return_conditional_losses_498742

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
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
с
c
*__inference_dropout_3_layer_call_fn_498959

inputs
identityИҐStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_497914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ыz
Р
C__inference_model_1_layer_call_and_return_conditional_losses_498542
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
&dense_4_matmul_readvariableop_resource:	А6
'dense_4_biasadd_readvariableop_resource:	АF
7batch_normalization_3_batchnorm_readvariableop_resource:	АJ
;batch_normalization_3_batchnorm_mul_readvariableop_resource:	АH
9batch_normalization_3_batchnorm_readvariableop_1_resource:	АH
9batch_normalization_3_batchnorm_readvariableop_2_resource:	А9
&dense_5_matmul_readvariableop_resource:	А@5
'dense_5_biasadd_readvariableop_resource:@E
7batch_normalization_4_batchnorm_readvariableop_resource:@I
;batch_normalization_4_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_4_batchnorm_readvariableop_1_resource:@G
9batch_normalization_4_batchnorm_readvariableop_2_resource:@8
&dense_6_matmul_readvariableop_resource:@@5
'dense_6_biasadd_readvariableop_resource:@E
7batch_normalization_5_batchnorm_readvariableop_resource:@I
;batch_normalization_5_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_5_batchnorm_readvariableop_1_resource:@G
9batch_normalization_5_batchnorm_readvariableop_2_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identityИҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ0batch_normalization_4/batchnorm/ReadVariableOp_1Ґ0batch_normalization_4/batchnorm/ReadVariableOp_2Ґ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐ.batch_normalization_5/batchnorm/ReadVariableOpҐ0batch_normalization_5/batchnorm/ReadVariableOp_1Ґ0batch_normalization_5/batchnorm/ReadVariableOp_2Ґ2batch_normalization_5/batchnorm/mul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOp[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
concatenate_1/concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0С
dense_4/MatMulMatMulconcatenate_1/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А£
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ї
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А}
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:АЂ
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ј
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А§
%batch_normalization_3/batchnorm/mul_1Muldense_4/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€АІ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0µ
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:АІ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0µ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аµ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А|
dropout_2/IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0О
dense_5/MatMulMatMuldropout_2/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ґ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:є
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:@™
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@£
%batch_normalization_4/batchnorm/mul_1Muldense_5/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0і
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:@¶
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0і
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@і
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@{
dropout_3/IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0О
dense_6/MatMulMatMuldropout_3/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ґ
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:є
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:@™
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@£
%batch_normalization_5/batchnorm/mul_1Muldense_6/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0і
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:@¶
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0і
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@і
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ь
dense_7/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ѓ
NoOpNoOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Q M
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
Є
ф
I__inference_concatenate_1_layer_call_and_return_conditional_losses_498722
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
ќ
∞
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497445

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
¶
—
6__inference_batch_normalization_4_layer_call_fn_498895

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497410o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
∆
Ъ
9__inference_transform_features_layer_layer_call_fn_497591
alcohol
	chlorides
placeholder
density
placeholder_1
placeholder_2
ph
placeholder_3
	sulphates
placeholder_4
placeholder_5
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10Р
PartitionedCallPartitionedCallalcohol	chloridesplaceholderdensityplaceholder_1placeholder_2phplaceholder_3	sulphatesplaceholder_4placeholder_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *з
_output_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497568`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_7IdentityPartitionedCall:output:7*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	alcohol:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	chlorides:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namecitric acid:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	density:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namefixed acidity:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namefree sulfur dioxide:KG
'
_output_shapes
:€€€€€€€€€

_user_specified_namepH:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresidual sugar:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	sulphates:]	Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nametotal sulfur dioxide:Y
U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namevolatile acidity
ќ
∞
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497363

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ж)
Ю
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497568

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
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€’
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6PlaceholderWithDefault:output:0inputs_7inputs_8inputs_9	inputs_10*
Tin
2	*
Tout
2	*ъ
_output_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_pruned_496913`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
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
£
F
*__inference_dropout_2_layer_call_fn_498827

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_497753a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_7_layer_call_and_return_conditional_losses_497824

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
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
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ю

х
C__inference_dense_5_layer_call_and_return_conditional_losses_498869

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Т%
к
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497492

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
®
—
6__inference_batch_normalization_4_layer_call_fn_498882

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497363o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ф+
Ў
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_499181
inputs_alcohol
inputs_chlorides
placeholder
inputs_density
placeholder_1
placeholder_2
	inputs_ph
placeholder_3
inputs_sulphates
placeholder_4
placeholder_5
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10C
ShapeShapeinputs_alcohol*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskE
Shape_1Shapeinputs_alcohol*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€П
PartitionedCallPartitionedCallinputs_alcoholinputs_chloridesplaceholderinputs_densityplaceholder_1placeholder_2	inputs_phPlaceholderWithDefault:output:0placeholder_3inputs_sulphatesplaceholder_4placeholder_5*
Tin
2	*
Tout
2	*ъ
_output_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_pruned_496913`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:W S
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs/alcohol:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinputs/chlorides:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs/citric acid:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs/density:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs/fixed acidity:c_
'
_output_shapes
:€€€€€€€€€
4
_user_specified_nameinputs/free sulfur dioxide:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs/pH:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameinputs/residual sugar:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinputs/sulphates:d	`
'
_output_shapes
:€€€€€€€€€
5
_user_specified_nameinputs/total sulfur dioxide:`
\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs/volatile acidity
р=
Ю

C__inference_model_1_layer_call_and_return_conditional_losses_497831

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
dense_4_497734:	А
dense_4_497736:	А+
batch_normalization_3_497739:	А+
batch_normalization_3_497741:	А+
batch_normalization_3_497743:	А+
batch_normalization_3_497745:	А!
dense_5_497767:	А@
dense_5_497769:@*
batch_normalization_4_497772:@*
batch_normalization_4_497774:@*
batch_normalization_4_497776:@*
batch_normalization_4_497778:@ 
dense_6_497800:@@
dense_6_497802:@*
batch_normalization_5_497805:@*
batch_normalization_5_497807:@*
batch_normalization_5_497809:@*
batch_normalization_5_497811:@ 
dense_7_497825:@
dense_7_497827:
identityИҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCall±
concatenate_1/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
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
GPU 2J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_497720Н
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_497734dense_4_497736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_497733З
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_497739batch_normalization_3_497741batch_normalization_3_497743batch_normalization_3_497745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497281л
dropout_2/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_497753И
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_5_497767dense_5_497769*
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
GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_497766Ж
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_497772batch_normalization_4_497774batch_normalization_4_497776batch_normalization_4_497778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497363к
dropout_3/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_497786И
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_6_497800dense_6_497802*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_497799Ж
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_497805batch_normalization_5_497807batch_normalization_5_497809batch_normalization_5_497811*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497445Ь
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_497825dense_7_497827*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_497824w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ё
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
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
ј
Х
(__inference_dense_7_layer_call_fn_499085

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallЎ
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_497824o
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
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
√
Ц
(__inference_dense_5_layer_call_fn_498858

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallЎ
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
GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_497766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_6_layer_call_and_return_conditional_losses_497799

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ѓ
’
6__inference_batch_normalization_3_layer_call_fn_498768

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497328p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
се
ґ$
"__inference__traced_restore_499556
file_prefix2
assignvariableop_dense_4_kernel:	А.
assignvariableop_1_dense_4_bias:	А=
.assignvariableop_2_batch_normalization_3_gamma:	А<
-assignvariableop_3_batch_normalization_3_beta:	АC
4assignvariableop_4_batch_normalization_3_moving_mean:	АG
8assignvariableop_5_batch_normalization_3_moving_variance:	А4
!assignvariableop_6_dense_5_kernel:	А@-
assignvariableop_7_dense_5_bias:@<
.assignvariableop_8_batch_normalization_4_gamma:@;
-assignvariableop_9_batch_normalization_4_beta:@C
5assignvariableop_10_batch_normalization_4_moving_mean:@G
9assignvariableop_11_batch_normalization_4_moving_variance:@4
"assignvariableop_12_dense_6_kernel:@@.
 assignvariableop_13_dense_6_bias:@=
/assignvariableop_14_batch_normalization_5_gamma:@<
.assignvariableop_15_batch_normalization_5_beta:@C
5assignvariableop_16_batch_normalization_5_moving_mean:@G
9assignvariableop_17_batch_normalization_5_moving_variance:@4
"assignvariableop_18_dense_7_kernel:@.
 assignvariableop_19_dense_7_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: <
)assignvariableop_29_adam_dense_4_kernel_m:	А6
'assignvariableop_30_adam_dense_4_bias_m:	АE
6assignvariableop_31_adam_batch_normalization_3_gamma_m:	АD
5assignvariableop_32_adam_batch_normalization_3_beta_m:	А<
)assignvariableop_33_adam_dense_5_kernel_m:	А@5
'assignvariableop_34_adam_dense_5_bias_m:@D
6assignvariableop_35_adam_batch_normalization_4_gamma_m:@C
5assignvariableop_36_adam_batch_normalization_4_beta_m:@;
)assignvariableop_37_adam_dense_6_kernel_m:@@5
'assignvariableop_38_adam_dense_6_bias_m:@D
6assignvariableop_39_adam_batch_normalization_5_gamma_m:@C
5assignvariableop_40_adam_batch_normalization_5_beta_m:@;
)assignvariableop_41_adam_dense_7_kernel_m:@5
'assignvariableop_42_adam_dense_7_bias_m:<
)assignvariableop_43_adam_dense_4_kernel_v:	А6
'assignvariableop_44_adam_dense_4_bias_v:	АE
6assignvariableop_45_adam_batch_normalization_3_gamma_v:	АD
5assignvariableop_46_adam_batch_normalization_3_beta_v:	А<
)assignvariableop_47_adam_dense_5_kernel_v:	А@5
'assignvariableop_48_adam_dense_5_bias_v:@D
6assignvariableop_49_adam_batch_normalization_4_gamma_v:@C
5assignvariableop_50_adam_batch_normalization_4_beta_v:@;
)assignvariableop_51_adam_dense_6_kernel_v:@@5
'assignvariableop_52_adam_dense_6_bias_v:@D
6assignvariableop_53_adam_batch_normalization_5_gamma_v:@C
5assignvariableop_54_adam_batch_normalization_5_beta_v:@;
)assignvariableop_55_adam_dense_7_kernel_v:@5
'assignvariableop_56_adam_dense_7_bias_v:
identity_58ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9—
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*ч
valueнBк:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHе
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*З
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B √
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapesл
и::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_3_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_3_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_3_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_4_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_4_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_5_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_5_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_5_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_3_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_3_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_5_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_5_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_4_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_4_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_6_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_6_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_5_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_5_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_7_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_7_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_3_gamma_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_3_beta_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_5_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_5_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_4_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_4_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_6_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_6_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_5_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_5_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_7_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_7_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 µ

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: Ґ

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*З
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Т%
к
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499076

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
№
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_497753

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ј
В
(__inference_model_1_layer_call_fn_498394
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
	inputs_10
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_497831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
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
ы	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_497947

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
а?
з

C__inference_model_1_layer_call_and_return_conditional_losses_498269
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
dense_4_498219:	А
dense_4_498221:	А+
batch_normalization_3_498224:	А+
batch_normalization_3_498226:	А+
batch_normalization_3_498228:	А+
batch_normalization_3_498230:	А!
dense_5_498234:	А@
dense_5_498236:@*
batch_normalization_4_498239:@*
batch_normalization_4_498241:@*
batch_normalization_4_498243:@*
batch_normalization_4_498245:@ 
dense_6_498249:@@
dense_6_498251:@*
batch_normalization_5_498254:@*
batch_normalization_5_498256:@*
batch_normalization_5_498258:@*
batch_normalization_5_498260:@ 
dense_7_498263:@
dense_7_498265:
identityИҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallъ
concatenate_1/PartitionedCallPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
GPU 2J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_497720Н
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_498219dense_4_498221*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_497733З
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_498224batch_normalization_3_498226batch_normalization_3_498228batch_normalization_3_498230*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497281л
dropout_2/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_497753И
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_5_498234dense_5_498236*
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
GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_497766Ж
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_498239batch_normalization_4_498241batch_normalization_4_498243batch_normalization_4_498245*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497363к
dropout_3/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_497786И
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_6_498249dense_6_498251*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_497799Ж
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_498254batch_normalization_5_498256batch_normalization_5_498258batch_normalization_5_498260*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497445Ь
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_498263dense_7_498265*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_497824w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ё
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Y U
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
Ю

х
C__inference_dense_5_layer_call_and_return_conditional_losses_497766

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ

ц
C__inference_dense_4_layer_call_and_return_conditional_losses_497733

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
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
у	
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_498976

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_7_layer_call_and_return_conditional_losses_499095

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
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
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Т%
к
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498949

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ы	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_498849

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
М
…
(__inference_model_1_layer_call_fn_497874
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

alcohol_xf
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_497831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
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
∞
’
6__inference_batch_normalization_3_layer_call_fn_498755

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497281p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≥
О
$__inference_signature_wrapper_496951

inputs
inputs_1
	inputs_10
	inputs_11
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7	
inputs_8
inputs_9
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7	

identity_8

identity_9
identity_10
identity_11њ
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2	*
Tout
2	*ъ
_output_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_pruned_496913`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_7IdentityPartitionedCall:output:7*
T0	*'
_output_shapes
:€€€€€€€€€b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_11IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*щ
_input_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_11:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_6:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_7:Q
M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_8:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_9
ё@
ж

C__inference_model_1_layer_call_and_return_conditional_losses_498107

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
dense_4_498057:	А
dense_4_498059:	А+
batch_normalization_3_498062:	А+
batch_normalization_3_498064:	А+
batch_normalization_3_498066:	А+
batch_normalization_3_498068:	А!
dense_5_498072:	А@
dense_5_498074:@*
batch_normalization_4_498077:@*
batch_normalization_4_498079:@*
batch_normalization_4_498081:@*
batch_normalization_4_498083:@ 
dense_6_498087:@@
dense_6_498089:@*
batch_normalization_5_498092:@*
batch_normalization_5_498094:@*
batch_normalization_5_498096:@*
batch_normalization_5_498098:@ 
dense_7_498101:@
dense_7_498103:
identityИҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCall±
concatenate_1/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
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
GPU 2J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_497720Н
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_4_498057dense_4_498059*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_497733Е
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_498062batch_normalization_3_498064batch_normalization_3_498066batch_normalization_3_498068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497328ы
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_497947Р
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_5_498072dense_5_498074*
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
GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_497766Д
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_4_498077batch_normalization_4_498079batch_normalization_4_498081batch_normalization_4_498083*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_497410Ю
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_497914Р
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_6_498087dense_6_498089*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_497799Д
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_5_498092batch_normalization_5_498094batch_normalization_5_498096batch_normalization_5_498098*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_497492Ь
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_7_498101dense_7_498103*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_497824w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:O K
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
Ў
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_498964

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
а
і
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498788

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€АЇ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ%
о
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497328

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	АИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ађ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Аі
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ак
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
џК
х
!__inference__wrapped_model_497257
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

alcohol_xfA
.model_1_dense_4_matmul_readvariableop_resource:	А>
/model_1_dense_4_biasadd_readvariableop_resource:	АN
?model_1_batch_normalization_3_batchnorm_readvariableop_resource:	АR
Cmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resource:	АP
Amodel_1_batch_normalization_3_batchnorm_readvariableop_1_resource:	АP
Amodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource:	АA
.model_1_dense_5_matmul_readvariableop_resource:	А@=
/model_1_dense_5_biasadd_readvariableop_resource:@M
?model_1_batch_normalization_4_batchnorm_readvariableop_resource:@Q
Cmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource:@O
Amodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource:@O
Amodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource:@@
.model_1_dense_6_matmul_readvariableop_resource:@@=
/model_1_dense_6_biasadd_readvariableop_resource:@M
?model_1_batch_normalization_5_batchnorm_readvariableop_resource:@Q
Cmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource:@O
Amodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource:@O
Amodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource:@@
.model_1_dense_7_matmul_readvariableop_resource:@=
/model_1_dense_7_biasadd_readvariableop_resource:
identityИҐ6model_1/batch_normalization_3/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ6model_1/batch_normalization_4/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOpҐ6model_1/batch_normalization_5/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOpҐ&model_1/dense_4/BiasAdd/ReadVariableOpҐ%model_1/dense_4/MatMul/ReadVariableOpҐ&model_1/dense_5/BiasAdd/ReadVariableOpҐ%model_1/dense_5/MatMul/ReadVariableOpҐ&model_1/dense_6/BiasAdd/ReadVariableOpҐ%model_1/dense_6/MatMul/ReadVariableOpҐ&model_1/dense_7/BiasAdd/ReadVariableOpҐ%model_1/dense_7/MatMul/ReadVariableOpc
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :≈
model_1/concatenate_1/concatConcatV2fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf*model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Х
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0©
model_1/dense_4/MatMulMatMul%model_1/concatenate_1/concat:output:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АУ
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0І
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аq
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А≥
6model_1/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0r
-model_1/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:“
+model_1/batch_normalization_3/batchnorm/addAddV2>model_1/batch_normalization_3/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:АН
-model_1/batch_normalization_3/batchnorm/RsqrtRsqrt/model_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:Аї
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0ѕ
+model_1/batch_normalization_3/batchnorm/mulMul1model_1/batch_normalization_3/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
-model_1/batch_normalization_3/batchnorm/mul_1Mul"model_1/dense_4/Relu:activations:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€АЈ
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
-model_1/batch_normalization_3/batchnorm/mul_2Mul@model_1/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:АЈ
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0Ќ
+model_1/batch_normalization_3/batchnorm/subSub@model_1/batch_normalization_3/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:АЌ
-model_1/batch_normalization_3/batchnorm/add_1AddV21model_1/batch_normalization_3/batchnorm/mul_1:z:0/model_1/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€АМ
model_1/dropout_2/IdentityIdentity1model_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€АХ
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0¶
model_1/dense_5/MatMulMatMul#model_1/dropout_2/Identity:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Т
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@p
model_1/dense_5/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
6model_1/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_1/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:—
+model_1/batch_normalization_4/batchnorm/addAddV2>model_1/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:@М
-model_1/batch_normalization_4/batchnorm/RsqrtRsqrt/model_1/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:@Ї
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
+model_1/batch_normalization_4/batchnorm/mulMul1model_1/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@ї
-model_1/batch_normalization_4/batchnorm/mul_1Mul"model_1/dense_5/Relu:activations:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
-model_1/batch_normalization_4/batchnorm/mul_2Mul@model_1/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:@ґ
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0ћ
+model_1/batch_normalization_4/batchnorm/subSub@model_1/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@ћ
-model_1/batch_normalization_4/batchnorm/add_1AddV21model_1/batch_normalization_4/batchnorm/mul_1:z:0/model_1/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
model_1/dropout_3/IdentityIdentity1model_1/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0¶
model_1/dense_6/MatMulMatMul#model_1/dropout_3/Identity:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Т
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@p
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
6model_1/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_1/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:—
+model_1/batch_normalization_5/batchnorm/addAddV2>model_1/batch_normalization_5/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:@М
-model_1/batch_normalization_5/batchnorm/RsqrtRsqrt/model_1/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:@Ї
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
+model_1/batch_normalization_5/batchnorm/mulMul1model_1/batch_normalization_5/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@ї
-model_1/batch_normalization_5/batchnorm/mul_1Mul"model_1/dense_6/Relu:activations:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
-model_1/batch_normalization_5/batchnorm/mul_2Mul@model_1/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:@ґ
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0ћ
+model_1/batch_normalization_5/batchnorm/subSub@model_1/batch_normalization_5/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@ћ
-model_1/batch_normalization_5/batchnorm/add_1AddV21model_1/batch_normalization_5/batchnorm/mul_1:z:0/model_1/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0і
model_1/dense_7/MatMulMatMul1model_1/batch_normalization_5/batchnorm/add_1:z:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€o
IdentityIdentity model_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ќ
NoOpNoOp7^model_1/batch_normalization_3/batchnorm/ReadVariableOp9^model_1/batch_normalization_3/batchnorm/ReadVariableOp_19^model_1/batch_normalization_3/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_4/batchnorm/ReadVariableOp9^model_1/batch_normalization_4/batchnorm/ReadVariableOp_19^model_1/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_5/batchnorm/ReadVariableOp9^model_1/batch_normalization_5/batchnorm/ReadVariableOp_19^model_1/batch_normalization_5/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 2p
6model_1/batch_normalization_3/batchnorm/ReadVariableOp6model_1/batch_normalization_3/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_18model_1/batch_normalization_3/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_28model_1/batch_normalization_3/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_4/batchnorm/ReadVariableOp6model_1/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_18model_1/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_28model_1/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_5/batchnorm/ReadVariableOp6model_1/batch_normalization_5/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_18model_1/batch_normalization_5/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_28model_1/batch_normalization_5/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp:Y U
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
ƒ
Ч
(__inference_dense_4_layer_call_fn_498731

inputs
unknown:	А
	unknown_0:	А
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_497733p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
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
тr
√
__inference__traced_save_499375
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
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
: ќ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*ч
valueнBк:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*З
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B №
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	Р
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

identity_1Identity_1:output:0*М
_input_shapesъ
ч: :	А:А:А:А:А:А:	А@:@:@:@:@:@:@@:@:@:@:@:@:@:: : : : : : : : : :	А:А:А:А:	А@:@:@:@:@@:@:@:@:@::	А:А:А:А:	А@:@:@:@:@@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А:!

_output_shapes	
:А:! 

_output_shapes	
:А:!!

_output_shapes	
:А:%"!

_output_shapes
:	А@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@:$& 

_output_shapes

:@@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::%,!

_output_shapes
:	А:!-

_output_shapes	
:А:!.

_output_shapes	
:А:!/

_output_shapes	
:А:%0!

_output_shapes
:	А@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:$4 

_output_shapes

:@@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:$8 

_output_shapes

:@: 9

_output_shapes
:::

_output_shapes
: 
±
В
(__inference_model_1_layer_call_fn_498449
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
	inputs_10
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identityИҐStatefulPartitionedCallґ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_498107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
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
Ж
…
(__inference_model_1_layer_call_fn_498205
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

alcohol_xf
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
	unknown_5:	А@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_498107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*О
_input_shapesь
щ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : 22
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
”*
µ
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497677
alcohol
	chlorides
placeholder
density
placeholder_1
placeholder_2
ph
placeholder_3
	sulphates
placeholder_4
placeholder_5
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10<
ShapeShapealcohol*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask>
Shape_1Shapealcohol*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€м
PartitionedCallPartitionedCallalcohol	chloridesplaceholderdensityplaceholder_1placeholder_2phPlaceholderWithDefault:output:0placeholder_3	sulphatesplaceholder_4placeholder_5*
Tin
2	*
Tout
2	*ъ
_output_shapesз
д:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_pruned_496913`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:€€€€€€€€€b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:€€€€€€€€€d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*ж
_input_shapes‘
—:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	alcohol:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	chlorides:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namecitric acid:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	density:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namefixed acidity:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_namefree sulfur dioxide:KG
'
_output_shapes
:€€€€€€€€€

_user_specified_namepH:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameresidual sugar:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	sulphates:]	Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nametotal sulfur dioxide:Y
U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namevolatile acidity"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*©
serving_defaultХ
9
examples-
serving_default_examples:0€€€€€€€€€<
output_00
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ьЄ
“
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
layer-14
layer_with_weights-2
layer-15
layer_with_weights-3
layer-16
layer-17
layer_with_weights-4
layer-18
layer_with_weights-5
layer-19
layer_with_weights-6
layer-20
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

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
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
к
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance"
_tf_keras_layer
Љ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
ї
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
к
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance"
_tf_keras_layer
Љ
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator"
_tf_keras_layer
ї
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
к
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance"
_tf_keras_layer
ї
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias"
_tf_keras_layer
Ћ
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
${ _saved_model_loader_tracked_dict"
_tf_keras_model
ґ
,0
-1
52
63
74
85
F6
G7
O8
P9
Q10
R11
`12
a13
i14
j15
k16
l17
s18
t19"
trackable_list_wrapper
Ж
,0
-1
52
63
F4
G5
O6
P7
`8
a9
i10
j11
s12
t13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ћ
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ё
Бtrace_0
Вtrace_1
Гtrace_2
Дtrace_32к
(__inference_model_1_layer_call_fn_497874
(__inference_model_1_layer_call_fn_498394
(__inference_model_1_layer_call_fn_498449
(__inference_model_1_layer_call_fn_498205њ
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
 zБtrace_0zВtrace_1zГtrace_2zДtrace_3
…
Еtrace_0
Жtrace_1
Зtrace_2
Иtrace_32÷
C__inference_model_1_layer_call_and_return_conditional_losses_498542
C__inference_model_1_layer_call_and_return_conditional_losses_498691
C__inference_model_1_layer_call_and_return_conditional_losses_498269
C__inference_model_1_layer_call_and_return_conditional_losses_498333њ
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
 zЕtrace_0zЖtrace_1zЗtrace_2zИtrace_3
щBц
!__inference__wrapped_model_497257fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
р
	Йiter
Кbeta_1
Лbeta_2

Мdecay
Нlearning_rate,mА-mБ5mВ6mГFmДGmЕOmЖPmЗ`mИamЙimКjmЛsmМtmН,vО-vП5vР6vСFvТGvУOvФPvХ`vЦavЧivШjvЩsvЪtvЫ"
	optimizer
-
Оserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ф
Фtrace_02’
.__inference_concatenate_1_layer_call_fn_498706Ґ
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
 zФtrace_0
П
Хtrace_02р
I__inference_concatenate_1_layer_call_and_return_conditional_losses_498722Ґ
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
 zХtrace_0
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
о
Ыtrace_02ѕ
(__inference_dense_4_layer_call_fn_498731Ґ
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
 zЫtrace_0
Й
Ьtrace_02к
C__inference_dense_4_layer_call_and_return_conditional_losses_498742Ґ
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
 zЬtrace_0
!:	А2dense_4/kernel
:А2dense_4/bias
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
б
Ґtrace_0
£trace_12¶
6__inference_batch_normalization_3_layer_call_fn_498755
6__inference_batch_normalization_3_layer_call_fn_498768≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zҐtrace_0z£trace_1
Ч
§trace_0
•trace_12№
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498788
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498822≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0z•trace_1
 "
trackable_list_wrapper
*:(А2batch_normalization_3/gamma
):'А2batch_normalization_3/beta
2:0А (2!batch_normalization_3/moving_mean
6:4А (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
…
Ђtrace_0
ђtrace_12О
*__inference_dropout_2_layer_call_fn_498827
*__inference_dropout_2_layer_call_fn_498832≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0zђtrace_1
€
≠trace_0
Ѓtrace_12ƒ
E__inference_dropout_2_layer_call_and_return_conditional_losses_498837
E__inference_dropout_2_layer_call_and_return_conditional_losses_498849≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≠trace_0zЃtrace_1
"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
о
іtrace_02ѕ
(__inference_dense_5_layer_call_fn_498858Ґ
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
 zіtrace_0
Й
µtrace_02к
C__inference_dense_5_layer_call_and_return_conditional_losses_498869Ґ
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
 zµtrace_0
!:	А@2dense_5/kernel
:@2dense_5/bias
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ґnon_trainable_variables
Јlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
б
їtrace_0
Љtrace_12¶
6__inference_batch_normalization_4_layer_call_fn_498882
6__inference_batch_normalization_4_layer_call_fn_498895≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zїtrace_0zЉtrace_1
Ч
љtrace_0
Њtrace_12№
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498915
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498949≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zљtrace_0zЊtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
…
ƒtrace_0
≈trace_12О
*__inference_dropout_3_layer_call_fn_498954
*__inference_dropout_3_layer_call_fn_498959≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zƒtrace_0z≈trace_1
€
∆trace_0
«trace_12ƒ
E__inference_dropout_3_layer_call_and_return_conditional_losses_498964
E__inference_dropout_3_layer_call_and_return_conditional_losses_498976≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∆trace_0z«trace_1
"
_generic_user_object
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
о
Ќtrace_02ѕ
(__inference_dense_6_layer_call_fn_498985Ґ
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
 zЌtrace_0
Й
ќtrace_02к
C__inference_dense_6_layer_call_and_return_conditional_losses_498996Ґ
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
 zќtrace_0
 :@@2dense_6/kernel
:@2dense_6/bias
<
i0
j1
k2
l3"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
б
‘trace_0
’trace_12¶
6__inference_batch_normalization_5_layer_call_fn_499009
6__inference_batch_normalization_5_layer_call_fn_499022≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0z’trace_1
Ч
÷trace_0
„trace_12№
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499042
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499076≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0z„trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
о
Ёtrace_02ѕ
(__inference_dense_7_layer_call_fn_499085Ґ
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
 zЁtrace_0
Й
ёtrace_02к
C__inference_dense_7_layer_call_and_return_conditional_losses_499095Ґ
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
 zёtrace_0
 :@2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
÷
дtrace_0
еtrace_12Ы
9__inference_transform_features_layer_layer_call_fn_497591
9__inference_transform_features_layer_layer_call_fn_499130Ґ
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
 zдtrace_0zеtrace_1
М
жtrace_0
зtrace_12—
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_499181
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497677Ґ
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
 zжtrace_0zзtrace_1
Ч
и	_imported
й_wrapped_function
к_structured_inputs
л_structured_outputs
м_output_to_inputs_map"
trackable_dict_wrapper
J
70
81
Q2
R3
k4
l5"
trackable_list_wrapper
∆
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
14
15
16
17
18
19
20
21"
trackable_list_wrapper
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ІB§
(__inference_model_1_layer_call_fn_497874fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
аBЁ
(__inference_model_1_layer_call_fn_498394inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
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
аBЁ
(__inference_model_1_layer_call_fn_498449inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
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
ІB§
(__inference_model_1_layer_call_fn_498205fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
ыBш
C__inference_model_1_layer_call_and_return_conditional_losses_498542inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
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
ыBш
C__inference_model_1_layer_call_and_return_conditional_losses_498691inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"њ
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
¬Bњ
C__inference_model_1_layer_call_and_return_conditional_losses_498269fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
¬Bњ
C__inference_model_1_layer_call_and_return_conditional_losses_498333fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
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
ћB…
$__inference_signature_wrapper_497163examples"Ф
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
…B∆
.__inference_concatenate_1_layer_call_fn_498706inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"Ґ
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
дBб
I__inference_concatenate_1_layer_call_and_return_conditional_losses_498722inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"Ґ
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
№Bў
(__inference_dense_4_layer_call_fn_498731inputs"Ґ
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
чBф
C__inference_dense_4_layer_call_and_return_conditional_losses_498742inputs"Ґ
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
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
6__inference_batch_normalization_3_layer_call_fn_498755inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
6__inference_batch_normalization_3_layer_call_fn_498768inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498788inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498822inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

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
пBм
*__inference_dropout_2_layer_call_fn_498827inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
*__inference_dropout_2_layer_call_fn_498832inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
КBЗ
E__inference_dropout_2_layer_call_and_return_conditional_losses_498837inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
КBЗ
E__inference_dropout_2_layer_call_and_return_conditional_losses_498849inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

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
№Bў
(__inference_dense_5_layer_call_fn_498858inputs"Ґ
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
чBф
C__inference_dense_5_layer_call_and_return_conditional_losses_498869inputs"Ґ
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
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
6__inference_batch_normalization_4_layer_call_fn_498882inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
6__inference_batch_normalization_4_layer_call_fn_498895inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498915inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498949inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

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
пBм
*__inference_dropout_3_layer_call_fn_498954inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
*__inference_dropout_3_layer_call_fn_498959inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
КBЗ
E__inference_dropout_3_layer_call_and_return_conditional_losses_498964inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
КBЗ
E__inference_dropout_3_layer_call_and_return_conditional_losses_498976inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

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
№Bў
(__inference_dense_6_layer_call_fn_498985inputs"Ґ
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
чBф
C__inference_dense_6_layer_call_and_return_conditional_losses_498996inputs"Ґ
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
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
6__inference_batch_normalization_5_layer_call_fn_499009inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
6__inference_batch_normalization_5_layer_call_fn_499022inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499042inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499076inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

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
№Bў
(__inference_dense_7_layer_call_fn_499085inputs"Ґ
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
чBф
C__inference_dense_7_layer_call_and_return_conditional_losses_499095inputs"Ґ
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
ъBч
9__inference_transform_features_layer_layer_call_fn_497591alcohol	chloridescitric aciddensityfixed acidityfree sulfur dioxidepHresidual sugar	sulphatestotal sulfur dioxidevolatile acidity"Ґ
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
«Bƒ
9__inference_transform_features_layer_layer_call_fn_499130inputs/alcoholinputs/chloridesinputs/citric acidinputs/densityinputs/fixed acidityinputs/free sulfur dioxide	inputs/pHinputs/residual sugarinputs/sulphatesinputs/total sulfur dioxideinputs/volatile acidity"Ґ
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
вBя
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_499181inputs/alcoholinputs/chloridesinputs/citric acidinputs/densityinputs/fixed acidityinputs/free sulfur dioxide	inputs/pHinputs/residual sugarinputs/sulphatesinputs/total sulfur dioxideinputs/volatile acidity"Ґ
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
ХBТ
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497677alcohol	chloridescitric aciddensityfixed acidityfree sulfur dioxidepHresidual sugar	sulphatestotal sulfur dioxidevolatile acidity"Ґ
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
»
пcreated_variables
р	resources
сtrackable_objects
тinitializers
уassets
ф
signatures
$х_self_saveable_object_factories
йtransform_fn"
_generic_user_object
ШBХ
__inference_pruned_496913inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
ц	variables
ч	keras_api

шtotal

щcount"
_tf_keras_metric
c
ъ	variables
ы	keras_api

ьtotal

эcount
ю
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
€serving_default"
signature_map
 "
trackable_dict_wrapper
0
ш0
щ1"
trackable_list_wrapper
.
ц	variables"
_generic_user_object
:  (2total
:  (2count
0
ь0
э1"
trackable_list_wrapper
.
ъ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
ЄBµ
$__inference_signature_wrapper_496951inputsinputs_1	inputs_10	inputs_11inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"Ф
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
&:$	А2Adam/dense_4/kernel/m
 :А2Adam/dense_4/bias/m
/:-А2"Adam/batch_normalization_3/gamma/m
.:,А2!Adam/batch_normalization_3/beta/m
&:$	А@2Adam/dense_5/kernel/m
:@2Adam/dense_5/bias/m
.:,@2"Adam/batch_normalization_4/gamma/m
-:+@2!Adam/batch_normalization_4/beta/m
%:#@@2Adam/dense_6/kernel/m
:@2Adam/dense_6/bias/m
.:,@2"Adam/batch_normalization_5/gamma/m
-:+@2!Adam/batch_normalization_5/beta/m
%:#@2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
&:$	А2Adam/dense_4/kernel/v
 :А2Adam/dense_4/bias/v
/:-А2"Adam/batch_normalization_3/gamma/v
.:,А2!Adam/batch_normalization_3/beta/v
&:$	А@2Adam/dense_5/kernel/v
:@2Adam/dense_5/bias/v
.:,@2"Adam/batch_normalization_4/gamma/v
-:+@2!Adam/batch_normalization_4/beta/v
%:#@@2Adam/dense_6/kernel/v
:@2Adam/dense_6/bias/v
.:,@2"Adam/batch_normalization_5/gamma/v
-:+@2!Adam/batch_normalization_5/beta/v
%:#@2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/vЁ
!__inference__wrapped_model_497257Ј,-8576FGROQP`alikjstлҐз
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
™ "1™.
,
dense_7!К
dense_7€€€€€€€€€є
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498788d85764Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ є
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_498822d78564Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ С
6__inference_batch_normalization_3_layer_call_fn_498755W85764Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€АС
6__inference_batch_normalization_3_layer_call_fn_498768W78564Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АЈ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498915bROQP3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ј
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_498949bQROP3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ П
6__inference_batch_normalization_4_layer_call_fn_498882UROQP3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@П
6__inference_batch_normalization_4_layer_call_fn_498895UQROP3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@Ј
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499042blikj3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ј
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_499076bklij3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ П
6__inference_batch_normalization_5_layer_call_fn_499009Ulikj3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@П
6__inference_batch_normalization_5_layer_call_fn_499022Uklij3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@Ь
I__inference_concatenate_1_layer_call_and_return_conditional_losses_498722ќ§Ґ†
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
Ъ ф
.__inference_concatenate_1_layer_call_fn_498706Ѕ§Ґ†
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
C__inference_dense_4_layer_call_and_return_conditional_losses_498742],-/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
(__inference_dense_4_layer_call_fn_498731P,-/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€А§
C__inference_dense_5_layer_call_and_return_conditional_losses_498869]FG0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
(__inference_dense_5_layer_call_fn_498858PFG0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@£
C__inference_dense_6_layer_call_and_return_conditional_losses_498996\`a/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ {
(__inference_dense_6_layer_call_fn_498985O`a/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@£
C__inference_dense_7_layer_call_and_return_conditional_losses_499095\st/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_7_layer_call_fn_499085Ost/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€І
E__inference_dropout_2_layer_call_and_return_conditional_losses_498837^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ І
E__inference_dropout_2_layer_call_and_return_conditional_losses_498849^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dropout_2_layer_call_fn_498827Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А
*__inference_dropout_2_layer_call_fn_498832Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€А•
E__inference_dropout_3_layer_call_and_return_conditional_losses_498964\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ •
E__inference_dropout_3_layer_call_and_return_conditional_losses_498976\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
*__inference_dropout_3_layer_call_fn_498954O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@}
*__inference_dropout_3_layer_call_fn_498959O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@ы
C__inference_model_1_layer_call_and_return_conditional_losses_498269≥,-8576FGROQP`alikjstуҐп
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
Ъ ы
C__inference_model_1_layer_call_and_return_conditional_losses_498333≥,-7856FGQROP`aklijstуҐп
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
Ъ і
C__inference_model_1_layer_call_and_return_conditional_losses_498542м,-8576FGROQP`alikjstђҐ®
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
Ъ і
C__inference_model_1_layer_call_and_return_conditional_losses_498691м,-7856FGQROP`aklijstђҐ®
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
Ъ ”
(__inference_model_1_layer_call_fn_497874¶,-8576FGROQP`alikjstуҐп
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
™ "К€€€€€€€€€”
(__inference_model_1_layer_call_fn_498205¶,-7856FGQROP`aklijstуҐп
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
™ "К€€€€€€€€€М
(__inference_model_1_layer_call_fn_498394я,-8576FGROQP`alikjstђҐ®
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
™ "К€€€€€€€€€М
(__inference_model_1_layer_call_fn_498449я,-7856FGQROP`aklijstђҐ®
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
™ "К€€€€€€€€€у
__inference_pruned_496913’чҐу
лҐз
д™а
3
alcohol(К%
inputs/alcohol€€€€€€€€€
7
	chlorides*К'
inputs/chlorides€€€€€€€€€
;
citric acid,К)
inputs/citric acid€€€€€€€€€
3
density(К%
inputs/density€€€€€€€€€
?
fixed acidity.К+
inputs/fixed acidity€€€€€€€€€
K
free sulfur dioxide4К1
inputs/free sulfur dioxide€€€€€€€€€
)
pH#К 
	inputs/pH€€€€€€€€€
3
quality(К%
inputs/quality€€€€€€€€€	
A
residual sugar/К,
inputs/residual sugar€€€€€€€€€
7
	sulphates*К'
inputs/sulphates€€€€€€€€€
M
total sulfur dioxide5К2
inputs/total sulfur dioxide€€€€€€€€€
E
volatile acidity1К.
inputs/volatile acidity€€€€€€€€€
™ "Ў™‘
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
2

quality_xf$К!

quality_xf€€€€€€€€€	
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
volatile_acidity_xf€€€€€€€€€„

$__inference_signature_wrapper_496951Ѓ
–Ґћ
Ґ 
ƒ™ј
*
inputs К
inputs€€€€€€€€€
.
inputs_1"К
inputs_1€€€€€€€€€
0
	inputs_10#К 
	inputs_10€€€€€€€€€
0
	inputs_11#К 
	inputs_11€€€€€€€€€
.
inputs_2"К
inputs_2€€€€€€€€€
.
inputs_3"К
inputs_3€€€€€€€€€
.
inputs_4"К
inputs_4€€€€€€€€€
.
inputs_5"К
inputs_5€€€€€€€€€
.
inputs_6"К
inputs_6€€€€€€€€€
.
inputs_7"К
inputs_7€€€€€€€€€	
.
inputs_8"К
inputs_8€€€€€€€€€
.
inputs_9"К
inputs_9€€€€€€€€€"Ў™‘
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
2

quality_xf$К!

quality_xf€€€€€€€€€	
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
volatile_acidity_xf€€€€€€€€€ѓ
$__inference_signature_wrapper_497163Ж,-8576FGROQP`alikjst9Ґ6
Ґ 
/™,
*
examplesК
examples€€€€€€€€€"3™0
.
output_0"К
output_0€€€€€€€€€Ъ
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_497677Ѕ
хҐс
йҐе
в™ё
,
alcohol!К
alcohol€€€€€€€€€
0
	chlorides#К 
	chlorides€€€€€€€€€
4
citric acid%К"
citric acid€€€€€€€€€
,
density!К
density€€€€€€€€€
8
fixed acidity'К$
fixed acidity€€€€€€€€€
D
free sulfur dioxide-К*
free sulfur dioxide€€€€€€€€€
"
pHК
pH€€€€€€€€€
:
residual sugar(К%
residual sugar€€€€€€€€€
0
	sulphates#К 
	sulphates€€€€€€€€€
F
total sulfur dioxide.К+
total sulfur dioxide€€€€€€€€€
>
volatile acidity*К'
volatile acidity€€€€€€€€€
™ "∆Ґ¬
Ї™ґ
4

alcohol_xf&К#
0/alcohol_xf€€€€€€€€€
8
chlorides_xf(К%
0/chlorides_xf€€€€€€€€€
<
citric_acid_xf*К'
0/citric_acid_xf€€€€€€€€€
4

density_xf&К#
0/density_xf€€€€€€€€€
@
fixed_acidity_xf,К)
0/fixed_acidity_xf€€€€€€€€€
L
free_sulfur_dioxide_xf2К/
0/free_sulfur_dioxide_xf€€€€€€€€€
*
ph_xf!К
0/ph_xf€€€€€€€€€
B
residual_sugar_xf-К*
0/residual_sugar_xf€€€€€€€€€
8
sulphates_xf(К%
0/sulphates_xf€€€€€€€€€
N
total_sulfur_dioxide_xf3К0
0/total_sulfur_dioxide_xf€€€€€€€€€
F
volatile_acidity_xf/К,
0/volatile_acidity_xf€€€€€€€€€
Ъ з
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_499181О¬ҐЊ
ґҐ≤
ѓ™Ђ
3
alcohol(К%
inputs/alcohol€€€€€€€€€
7
	chlorides*К'
inputs/chlorides€€€€€€€€€
;
citric acid,К)
inputs/citric acid€€€€€€€€€
3
density(К%
inputs/density€€€€€€€€€
?
fixed acidity.К+
inputs/fixed acidity€€€€€€€€€
K
free sulfur dioxide4К1
inputs/free sulfur dioxide€€€€€€€€€
)
pH#К 
	inputs/pH€€€€€€€€€
A
residual sugar/К,
inputs/residual sugar€€€€€€€€€
7
	sulphates*К'
inputs/sulphates€€€€€€€€€
M
total sulfur dioxide5К2
inputs/total sulfur dioxide€€€€€€€€€
E
volatile acidity1К.
inputs/volatile acidity€€€€€€€€€
™ "∆Ґ¬
Ї™ґ
4

alcohol_xf&К#
0/alcohol_xf€€€€€€€€€
8
chlorides_xf(К%
0/chlorides_xf€€€€€€€€€
<
citric_acid_xf*К'
0/citric_acid_xf€€€€€€€€€
4

density_xf&К#
0/density_xf€€€€€€€€€
@
fixed_acidity_xf,К)
0/fixed_acidity_xf€€€€€€€€€
L
free_sulfur_dioxide_xf2К/
0/free_sulfur_dioxide_xf€€€€€€€€€
*
ph_xf!К
0/ph_xf€€€€€€€€€
B
residual_sugar_xf-К*
0/residual_sugar_xf€€€€€€€€€
8
sulphates_xf(К%
0/sulphates_xf€€€€€€€€€
N
total_sulfur_dioxide_xf3К0
0/total_sulfur_dioxide_xf€€€€€€€€€
F
volatile_acidity_xf/К,
0/volatile_acidity_xf€€€€€€€€€
Ъ Ё

9__inference_transform_features_layer_layer_call_fn_497591Я
хҐс
йҐе
в™ё
,
alcohol!К
alcohol€€€€€€€€€
0
	chlorides#К 
	chlorides€€€€€€€€€
4
citric acid%К"
citric acid€€€€€€€€€
,
density!К
density€€€€€€€€€
8
fixed acidity'К$
fixed acidity€€€€€€€€€
D
free sulfur dioxide-К*
free sulfur dioxide€€€€€€€€€
"
pHК
pH€€€€€€€€€
:
residual sugar(К%
residual sugar€€€€€€€€€
0
	sulphates#К 
	sulphates€€€€€€€€€
F
total sulfur dioxide.К+
total sulfur dioxide€€€€€€€€€
>
volatile acidity*К'
volatile acidity€€€€€€€€€
™ "§™†
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
volatile_acidity_xf€€€€€€€€€™
9__inference_transform_features_layer_layer_call_fn_499130м
¬ҐЊ
ґҐ≤
ѓ™Ђ
3
alcohol(К%
inputs/alcohol€€€€€€€€€
7
	chlorides*К'
inputs/chlorides€€€€€€€€€
;
citric acid,К)
inputs/citric acid€€€€€€€€€
3
density(К%
inputs/density€€€€€€€€€
?
fixed acidity.К+
inputs/fixed acidity€€€€€€€€€
K
free sulfur dioxide4К1
inputs/free sulfur dioxide€€€€€€€€€
)
pH#К 
	inputs/pH€€€€€€€€€
A
residual sugar/К,
inputs/residual sugar€€€€€€€€€
7
	sulphates*К'
inputs/sulphates€€€€€€€€€
M
total sulfur dioxide5К2
inputs/total sulfur dioxide€€€€€€€€€
E
volatile acidity1К.
inputs/volatile acidity€€€€€€€€€
™ "§™†
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
volatile_acidity_xf€€€€€€€€€