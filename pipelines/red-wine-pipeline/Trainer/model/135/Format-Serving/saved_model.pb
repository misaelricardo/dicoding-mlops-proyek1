ОЅ
бш
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(љ
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
љ
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
│
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
dtypetypeѕ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0ѕ
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
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
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
-
Sqrt
x"T
y"T"
Ttype:

2
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.12v2.10.0-76-gfdfc646704c8ь 
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *кS@
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *йѓ6
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *к-?
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *ЬєD
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *L:B
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *«¤B
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *A
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *rw;
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *з▓=
L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *Фн@
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *а«%@
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *х=
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *╗▀Љ>
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *Ю/=
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *JР?
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *vаL@
M
Const_16Const*
_output_shapes
: *
dtype0*
valueB
 *$A
M
Const_17Const*
_output_shapes
: *
dtype0*
valueB
 *_=њ?
M
Const_18Const*
_output_shapes
: *
dtype0*
valueB
 *<'A
M
Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *Ї?ш<
M
Const_20Const*
_output_shapes
: *
dtype0*
valueB
 *Ё¤)?
M
Const_21Const*
_output_shapes
: *
dtype0*
valueB
 *│p╩<
ђ
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_11/kernel/v
Ђ
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:@*
dtype0
џ
!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_8/beta/v
Њ
5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes
:@*
dtype0
ю
"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_8/gamma/v
Ћ
6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes
:@*
dtype0
ђ
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:@*
dtype0
ѕ
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_10/kernel/v
Ђ
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:@@*
dtype0
џ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/v
Њ
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:@*
dtype0
ю
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/v
Ћ
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:@*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:@*
dtype0
Є
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*&
shared_nameAdam/dense_9/kernel/v
ђ
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	ђ@*
dtype0
Џ
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_6/beta/v
ћ
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_6/gamma/v
ќ
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes	
:ђ*
dtype0

Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_8/bias/v
x
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_nameAdam/dense_8/kernel/v
ђ
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_11/kernel/m
Ђ
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:@*
dtype0
џ
!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_8/beta/m
Њ
5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes
:@*
dtype0
ю
"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_8/gamma/m
Ћ
6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes
:@*
dtype0
ђ
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:@*
dtype0
ѕ
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_10/kernel/m
Ђ
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:@@*
dtype0
џ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/m
Њ
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:@*
dtype0
ю
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/m
Ћ
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:@*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:@*
dtype0
Є
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*&
shared_nameAdam/dense_9/kernel/m
ђ
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	ђ@*
dtype0
Џ
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_6/beta/m
ћ
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_6/gamma/m
ќ
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes	
:ђ*
dtype0

Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_8/bias/m
x
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_nameAdam/dense_8/kernel/m
ђ
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	ђ*
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
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:@*
dtype0
б
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_8/moving_variance
Џ
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:@*
dtype0
џ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_8/moving_mean
Њ
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:@*
dtype0
ї
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_8/beta
Ё
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:@*
dtype0
ј
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_8/gamma
Є
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:@*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:@*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:@@*
dtype0
б
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
Џ
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
џ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
Њ
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
ї
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
Ё
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
ј
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
Є
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:@*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	ђ@*
dtype0
Б
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_6/moving_variance
ю
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_6/moving_mean
ћ
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_6/beta
є
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_6/gamma
ѕ
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:ђ*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:ђ*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	ђ*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
к
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesConst_16Const_15Const_14Const_13Const_12Const_11Const_10Const_9Const_8Const_7Const_6Const_5Const_4Const_3Const_2Const_1ConstConst_21Const_20Const_19Const_18Const_17dense_8/kerneldense_8/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/betadense_9/kerneldense_9/bias%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/betadense_10/kerneldense_10/bias%batch_normalization_8/moving_variancebatch_normalization_8/gamma!batch_normalization_8/moving_meanbatch_normalization_8/betadense_11/kerneldense_11/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_783263

NoOpNoOp
ОЉ
Const_22Const"/device:CPU:0*
_output_shapes
: *
dtype0*јЉ
valueЃЉB љ Bэљ
╗
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
ј
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
д
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
Н
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
Ц
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator* 
д
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
Н
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
Ц
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator* 
д
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
Н
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
д
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias*
┤
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
${ _saved_model_loader_tracked_dict* 
џ
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
▒
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
ђlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Ђtrace_0
ѓtrace_1
Ѓtrace_2
ёtrace_3* 
:
Ёtrace_0
єtrace_1
Єtrace_2
ѕtrace_3* 
* 
р
	Ѕiter
іbeta_1
Іbeta_2

їdecay
Їlearning_rate,mќ-mЌ5mў6mЎFmџGmЏOmюPmЮ`mъamЪimаjmАsmбtmБ,vц-vЦ5vд6vДFvеGvЕOvфPvФ`vгavГiv«jv»sv░tv▒*

јserving_default* 
* 
* 
* 
ќ
Јnon_trainable_variables
љlayers
Љmetrics
 њlayer_regularization_losses
Њlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

ћtrace_0* 

Ћtrace_0* 

,0
-1*

,0
-1*
* 
ў
ќnon_trainable_variables
Ќlayers
ўmetrics
 Ўlayer_regularization_losses
џlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Џtrace_0* 

юtrace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
50
61
72
83*

50
61*
* 
ў
Юnon_trainable_variables
ъlayers
Ъmetrics
 аlayer_regularization_losses
Аlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

бtrace_0
Бtrace_1* 

цtrace_0
Цtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

Фtrace_0
гtrace_1* 

Гtrace_0
«trace_1* 
* 

F0
G1*

F0
G1*
* 
ў
»non_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

┤trace_0* 

хtrace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
O0
P1
Q2
R3*

O0
P1*
* 
ў
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

╗trace_0
╝trace_1* 

йtrace_0
Йtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

─trace_0
┼trace_1* 

кtrace_0
Кtrace_1* 
* 

`0
a1*

`0
a1*
* 
ў
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

═trace_0* 

╬trace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
i0
j1
k2
l3*

i0
j1*
* 
ў
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

нtrace_0
Нtrace_1* 

оtrace_0
Оtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

s0
t1*

s0
t1*
* 
ў
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

Пtrace_0* 

яtrace_0* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

Сtrace_0
тtrace_1* 

Тtrace_0
уtrace_1* 
y
У	_imported
ж_wrapped_function
Ж_structured_inputs
в_structured_outputs
В_output_to_inputs_map* 
.
70
81
Q2
R3
k4
l5*
ф
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
ь0
Ь1*
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
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
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
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
г
Ёcreated_variables
є	resources
Єtrackable_objects
ѕinitializers
Ѕassets
і
signatures
$І_self_saveable_object_factories
жtransform_fn* 
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
* 
* 
* 
<
ї	variables
Ї	keras_api

јtotal

Јcount*
M
љ	variables
Љ	keras_api

њtotal

Њcount
ћ
_fn_kwargs*
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

Ћserving_default* 
* 

ј0
Ј1*

ї	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

њ0
Њ1*

љ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ь
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21* 
Ђ{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ђ{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_8/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ђ{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ђ{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_8/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
К
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpConst_22*F
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
GPU 2J 8ѓ *(
f#R!
__inference__traced_save_785761
╦
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancedense_9/kerneldense_9/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_10/kerneldense_10/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_8/kernel/mAdam/dense_8/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/dense_9/kernel/mAdam/dense_9/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/dense_10/kernel/mAdam/dense_10/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/dense_9/kernel/vAdam/dense_9/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/dense_10/kernel/vAdam/dense_10/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/vAdam/dense_11/kernel/vAdam/dense_11/bias/v*E
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_785942ж╗
СB
х
C__inference_model_2_layer_call_and_return_conditional_losses_784609
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
dense_8_784559:	ђ
dense_8_784561:	ђ+
batch_normalization_6_784564:	ђ+
batch_normalization_6_784566:	ђ+
batch_normalization_6_784568:	ђ+
batch_normalization_6_784570:	ђ!
dense_9_784574:	ђ@
dense_9_784576:@*
batch_normalization_7_784579:@*
batch_normalization_7_784581:@*
batch_normalization_7_784583:@*
batch_normalization_7_784585:@!
dense_10_784589:@@
dense_10_784591:@*
batch_normalization_8_784594:@*
batch_normalization_8_784596:@*
batch_normalization_8_784598:@*
batch_normalization_8_784600:@!
dense_11_784603:@
dense_11_784605:
identityѕб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallб!dropout_4/StatefulPartitionedCallб!dropout_5/StatefulPartitionedCallЩ
concatenate_2/PartitionedCallPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_783996Ї
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_784559dense_8_784561*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_784009Ё
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_784564batch_normalization_6_784566batch_normalization_6_784568batch_normalization_6_784570*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783428ч
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_784223љ
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_9_784574dense_9_784576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_784042ё
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_784579batch_normalization_7_784581batch_normalization_7_784583batch_normalization_7_784585*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783510ъ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_784190ћ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_10_784589dense_10_784591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_784075Ё
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_8_784594batch_normalization_8_784596batch_normalization_8_784598batch_normalization_8_784600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783592а
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_11_784603dense_11_784605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_784100x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         е
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:Y U
'
_output_shapes
:         
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:         
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:         
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:         
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:         
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:         
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:         
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:         
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:         

_user_specified_nameph_xf:U	Q
'
_output_shapes
:         
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:         
$
_user_specified_name
alcohol_xf
ч	
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_784223

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
О(
р
$__inference_signature_wrapper_782897

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
inputs_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
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
identity_11л
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*-
Tin&
$2"	*
Tout
2	*Щ
_output_shapesу
С:         :         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *"
fR
__inference_pruned_782837`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:         b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:         b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:         b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:         b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:         b

Identity_7IdentityPartitionedCall:output:7*
T0	*'
_output_shapes
:         b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:         b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:         d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:         d
Identity_11IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*Ц
_input_shapesЊ
љ:         :         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:         
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:         
#
_user_specified_name	inputs_11:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_6:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs_7:Q
M
'
_output_shapes
:         
"
_user_specified_name
inputs_8:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_9:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 
ъ

ш
C__inference_dense_9_layer_call_and_return_conditional_losses_784042

inputs1
matmul_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
њ%
Ж
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783592

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:@Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         @h
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
з	
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_785252

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ь'
№
;__inference_transform_features_layer_1_layer_call_fn_783779
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
placeholder_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
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
identity_10Б
PartitionedCallPartitionedCallalcohol	chloridesplaceholderdensityplaceholder_1placeholder_2phplaceholder_3	sulphatesplaceholder_4placeholder_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *у
_output_shapesн
Л:         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783712`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:         b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:         b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:         b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:         b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:         b

Identity_7IdentityPartitionedCall:output:7*
T0*'
_output_shapes
:         b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:         b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:         d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*њ
_input_shapesђ
§:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :P L
'
_output_shapes
:         
!
_user_specified_name	alcohol:RN
'
_output_shapes
:         
#
_user_specified_name	chlorides:TP
'
_output_shapes
:         
%
_user_specified_namecitric acid:PL
'
_output_shapes
:         
!
_user_specified_name	density:VR
'
_output_shapes
:         
'
_user_specified_namefixed acidity:\X
'
_output_shapes
:         
-
_user_specified_namefree sulfur dioxide:KG
'
_output_shapes
:         

_user_specified_namepH:WS
'
_output_shapes
:         
(
_user_specified_nameresidual sugar:RN
'
_output_shapes
:         
#
_user_specified_name	sulphates:]	Y
'
_output_shapes
:         
.
_user_specified_nametotal sulfur dioxide:Y
U
'
_output_shapes
:         
*
_user_specified_namevolatile acidity:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: 
щ4
і
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783953
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
placeholder_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
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
valueB:Л
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
valueB:█
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
:         ћ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:         *
dtype0	*
shape:         §
PartitionedCallPartitionedCallalcohol	chloridesplaceholderdensityplaceholder_1placeholder_2phPlaceholderWithDefault:output:0placeholder_3	sulphatesplaceholder_4placeholder_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*-
Tin&
$2"	*
Tout
2	*Щ
_output_shapesу
С:         :         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *"
fR
__inference_pruned_782837`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:         b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:         b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:         b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:         b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:         b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:         b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:         c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:         d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*њ
_input_shapesђ
§:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :P L
'
_output_shapes
:         
!
_user_specified_name	alcohol:RN
'
_output_shapes
:         
#
_user_specified_name	chlorides:TP
'
_output_shapes
:         
%
_user_specified_namecitric acid:PL
'
_output_shapes
:         
!
_user_specified_name	density:VR
'
_output_shapes
:         
'
_user_specified_namefixed acidity:\X
'
_output_shapes
:         
-
_user_specified_namefree sulfur dioxide:KG
'
_output_shapes
:         

_user_specified_namepH:WS
'
_output_shapes
:         
(
_user_specified_nameresidual sugar:RN
'
_output_shapes
:         
#
_user_specified_name	sulphates:]	Y
'
_output_shapes
:         
.
_user_specified_nametotal sulfur dioxide:Y
U
'
_output_shapes
:         
*
_user_specified_namevolatile acidity:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: 
з	
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_784190

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ъ

ш
C__inference_dense_9_layer_call_and_return_conditional_losses_785145

inputs1
matmul_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
То
╩
C__inference_model_2_layer_call_and_return_conditional_losses_784967
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
&dense_8_matmul_readvariableop_resource:	ђ6
'dense_8_biasadd_readvariableop_resource:	ђL
=batch_normalization_6_assignmovingavg_readvariableop_resource:	ђN
?batch_normalization_6_assignmovingavg_1_readvariableop_resource:	ђJ
;batch_normalization_6_batchnorm_mul_readvariableop_resource:	ђF
7batch_normalization_6_batchnorm_readvariableop_resource:	ђ9
&dense_9_matmul_readvariableop_resource:	ђ@5
'dense_9_biasadd_readvariableop_resource:@K
=batch_normalization_7_assignmovingavg_readvariableop_resource:@M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_7_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_7_batchnorm_readvariableop_resource:@9
'dense_10_matmul_readvariableop_resource:@@6
(dense_10_biasadd_readvariableop_resource:@K
=batch_normalization_8_assignmovingavg_readvariableop_resource:@M
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_8_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_8_batchnorm_readvariableop_resource:@9
'dense_11_matmul_readvariableop_resource:@6
(dense_11_biasadd_readvariableop_resource:
identityѕб%batch_normalization_6/AssignMovingAvgб4batch_normalization_6/AssignMovingAvg/ReadVariableOpб'batch_normalization_6/AssignMovingAvg_1б6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_6/batchnorm/ReadVariableOpб2batch_normalization_6/batchnorm/mul/ReadVariableOpб%batch_normalization_7/AssignMovingAvgб4batch_normalization_7/AssignMovingAvg/ReadVariableOpб'batch_normalization_7/AssignMovingAvg_1б6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_7/batchnorm/ReadVariableOpб2batch_normalization_7/batchnorm/mul/ReadVariableOpб%batch_normalization_8/AssignMovingAvgб4batch_normalization_8/AssignMovingAvg/ReadVariableOpб'batch_normalization_8/AssignMovingAvg_1б6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpб.batch_normalization_8/batchnorm/ReadVariableOpб2batch_normalization_8/batchnorm/mul/ReadVariableOpбdense_10/BiasAdd/ReadVariableOpбdense_10/MatMul/ReadVariableOpбdense_11/BiasAdd/ReadVariableOpбdense_11/MatMul/ReadVariableOpбdense_8/BiasAdd/ReadVariableOpбdense_8/MatMul/ReadVariableOpбdense_9/BiasAdd/ReadVariableOpбdense_9/MatMul/ReadVariableOp[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ь
concatenate_2/concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ё
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Љ
dense_8/MatMulMatMulconcatenate_2/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ј
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ~
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: └
"batch_normalization_6/moments/meanMeandense_8/Relu:activations:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(Љ
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	ђ╚
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђѓ
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: р
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(џ
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 а
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 p
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<»
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype0─
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes	
:ђ╗
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђё
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<│
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0╩
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ┴
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђї
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:┤
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ}
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:ђФ
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђц
%batch_normalization_6/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђФ
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђБ
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0│
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђх
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ю
dropout_4/dropout/MulMul)batch_normalization_6/batchnorm/add_1:z:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:         ђp
dropout_4/dropout/ShapeShape)batch_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:А
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>┼
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђё
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђѕ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђЁ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0ј
dense_9/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ј
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         @~
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┐
"batch_normalization_7/moments/meanMeandense_9/Relu:activations:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(љ
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:@К
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @ѓ
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(Ў
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Ъ
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<«
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0├
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
:@║
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ё
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<▓
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0╔
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@└
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@ї
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:│
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@ф
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Х
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Б
%batch_normalization_7/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @ф
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@б
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0▓
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@┤
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Џ
dropout_5/dropout/MulMul)batch_normalization_7/batchnorm/add_1:z:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:         @p
dropout_5/dropout/ShapeShape)batch_normalization_7/batchnorm/add_1:z:0*
T0*
_output_shapes
:а
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>─
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @Ѓ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @Є
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:         @є
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0љ
dense_10/MatMulMatMuldropout_5/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ё
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:         @~
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: └
"batch_normalization_8/moments/meanMeandense_10/Relu:activations:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(љ
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:@╚
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:03batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @ѓ
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(Ў
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Ъ
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<«
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0├
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes
:@║
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ё
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<▓
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0╔
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@└
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@ї
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:│
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:@ф
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Х
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@ц
%batch_normalization_8/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @ф
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:@б
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0▓
#batch_normalization_8/batchnorm/subSub6batch_normalization_8/batchnorm/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@┤
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @є
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0ъ
dense_11/MatMulMatMul)batch_normalization_8/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         к	
NoOpNoOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp3^batch_normalization_8/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_6/AssignMovingAvg%batch_normalization_6/AssignMovingAvg2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_6/AssignMovingAvg_1'batch_normalization_6/AssignMovingAvg_12p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2N
%batch_normalization_7/AssignMovingAvg%batch_normalization_7/AssignMovingAvg2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_7/AssignMovingAvg_1'batch_normalization_7/AssignMovingAvg_12p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2N
%batch_normalization_8/AssignMovingAvg%batch_normalization_8/AssignMovingAvg2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_8/AssignMovingAvg_1'batch_normalization_8/AssignMovingAvg_12p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10
Б
F
*__inference_dropout_4_layer_call_fn_785103

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_784029a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
░
Н
6__inference_batch_normalization_6_layer_call_fn_785031

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783381p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┼z
ў
C__inference_model_2_layer_call_and_return_conditional_losses_784818
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
&dense_8_matmul_readvariableop_resource:	ђ6
'dense_8_biasadd_readvariableop_resource:	ђF
7batch_normalization_6_batchnorm_readvariableop_resource:	ђJ
;batch_normalization_6_batchnorm_mul_readvariableop_resource:	ђH
9batch_normalization_6_batchnorm_readvariableop_1_resource:	ђH
9batch_normalization_6_batchnorm_readvariableop_2_resource:	ђ9
&dense_9_matmul_readvariableop_resource:	ђ@5
'dense_9_biasadd_readvariableop_resource:@E
7batch_normalization_7_batchnorm_readvariableop_resource:@I
;batch_normalization_7_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_7_batchnorm_readvariableop_1_resource:@G
9batch_normalization_7_batchnorm_readvariableop_2_resource:@9
'dense_10_matmul_readvariableop_resource:@@6
(dense_10_biasadd_readvariableop_resource:@E
7batch_normalization_8_batchnorm_readvariableop_resource:@I
;batch_normalization_8_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_8_batchnorm_readvariableop_1_resource:@G
9batch_normalization_8_batchnorm_readvariableop_2_resource:@9
'dense_11_matmul_readvariableop_resource:@6
(dense_11_biasadd_readvariableop_resource:
identityѕб.batch_normalization_6/batchnorm/ReadVariableOpб0batch_normalization_6/batchnorm/ReadVariableOp_1б0batch_normalization_6/batchnorm/ReadVariableOp_2б2batch_normalization_6/batchnorm/mul/ReadVariableOpб.batch_normalization_7/batchnorm/ReadVariableOpб0batch_normalization_7/batchnorm/ReadVariableOp_1б0batch_normalization_7/batchnorm/ReadVariableOp_2б2batch_normalization_7/batchnorm/mul/ReadVariableOpб.batch_normalization_8/batchnorm/ReadVariableOpб0batch_normalization_8/batchnorm/ReadVariableOp_1б0batch_normalization_8/batchnorm/ReadVariableOp_2б2batch_normalization_8/batchnorm/mul/ReadVariableOpбdense_10/BiasAdd/ReadVariableOpбdense_10/MatMul/ReadVariableOpбdense_11/BiasAdd/ReadVariableOpбdense_11/MatMul/ReadVariableOpбdense_8/BiasAdd/ReadVariableOpбdense_8/MatMul/ReadVariableOpбdense_9/BiasAdd/ReadVariableOpбdense_9/MatMul/ReadVariableOp[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ь
concatenate_2/concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ё
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Љ
dense_8/MatMulMatMulconcatenate_2/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ј
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђБ
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0j
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:║
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ}
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:ђФ
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђц
%batch_normalization_6/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђД
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0х
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђД
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0х
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђх
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ|
dropout_4/IdentityIdentity)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђЁ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0ј
dense_9/MatMulMatMuldropout_4/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ѓ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ј
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         @б
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:╣
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@ф
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Х
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Б
%batch_normalization_7/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @д
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0┤
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@д
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0┤
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@┤
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @{
dropout_5/IdentityIdentity)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         @є
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0љ
dense_10/MatMulMatMuldropout_5/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ё
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:         @б
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:╣
#batch_normalization_8/batchnorm/addAddV26batch_normalization_8/batchnorm/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:@ф
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Х
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@ц
%batch_normalization_8/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @д
0batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0┤
%batch_normalization_8/batchnorm/mul_2Mul8batch_normalization_8/batchnorm/ReadVariableOp_1:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:@д
0batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0┤
#batch_normalization_8/batchnorm/subSub8batch_normalization_8/batchnorm/ReadVariableOp_2:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@┤
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @є
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0ъ
dense_11/MatMulMatMul)batch_normalization_8/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ▓
NoOpNoOp/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp1^batch_normalization_8/batchnorm/ReadVariableOp_11^batch_normalization_8/batchnorm/ReadVariableOp_23^batch_normalization_8/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2d
0batch_normalization_6/batchnorm/ReadVariableOp_10batch_normalization_6/batchnorm/ReadVariableOp_12d
0batch_normalization_6/batchnorm/ReadVariableOp_20batch_normalization_6/batchnorm/ReadVariableOp_22h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2d
0batch_normalization_7/batchnorm/ReadVariableOp_10batch_normalization_7/batchnorm/ReadVariableOp_12d
0batch_normalization_7/batchnorm/ReadVariableOp_20batch_normalization_7/batchnorm/ReadVariableOp_22h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2d
0batch_normalization_8/batchnorm/ReadVariableOp_10batch_normalization_8/batchnorm/ReadVariableOp_12d
0batch_normalization_8/batchnorm/ReadVariableOp_20batch_normalization_8/batchnorm/ReadVariableOp_22h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10
Ъ
F
*__inference_dropout_5_layer_call_fn_785230

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_784062`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ЅТ
┬$
"__inference__traced_restore_785942
file_prefix2
assignvariableop_dense_8_kernel:	ђ.
assignvariableop_1_dense_8_bias:	ђ=
.assignvariableop_2_batch_normalization_6_gamma:	ђ<
-assignvariableop_3_batch_normalization_6_beta:	ђC
4assignvariableop_4_batch_normalization_6_moving_mean:	ђG
8assignvariableop_5_batch_normalization_6_moving_variance:	ђ4
!assignvariableop_6_dense_9_kernel:	ђ@-
assignvariableop_7_dense_9_bias:@<
.assignvariableop_8_batch_normalization_7_gamma:@;
-assignvariableop_9_batch_normalization_7_beta:@C
5assignvariableop_10_batch_normalization_7_moving_mean:@G
9assignvariableop_11_batch_normalization_7_moving_variance:@5
#assignvariableop_12_dense_10_kernel:@@/
!assignvariableop_13_dense_10_bias:@=
/assignvariableop_14_batch_normalization_8_gamma:@<
.assignvariableop_15_batch_normalization_8_beta:@C
5assignvariableop_16_batch_normalization_8_moving_mean:@G
9assignvariableop_17_batch_normalization_8_moving_variance:@5
#assignvariableop_18_dense_11_kernel:@/
!assignvariableop_19_dense_11_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: <
)assignvariableop_29_adam_dense_8_kernel_m:	ђ6
'assignvariableop_30_adam_dense_8_bias_m:	ђE
6assignvariableop_31_adam_batch_normalization_6_gamma_m:	ђD
5assignvariableop_32_adam_batch_normalization_6_beta_m:	ђ<
)assignvariableop_33_adam_dense_9_kernel_m:	ђ@5
'assignvariableop_34_adam_dense_9_bias_m:@D
6assignvariableop_35_adam_batch_normalization_7_gamma_m:@C
5assignvariableop_36_adam_batch_normalization_7_beta_m:@<
*assignvariableop_37_adam_dense_10_kernel_m:@@6
(assignvariableop_38_adam_dense_10_bias_m:@D
6assignvariableop_39_adam_batch_normalization_8_gamma_m:@C
5assignvariableop_40_adam_batch_normalization_8_beta_m:@<
*assignvariableop_41_adam_dense_11_kernel_m:@6
(assignvariableop_42_adam_dense_11_bias_m:<
)assignvariableop_43_adam_dense_8_kernel_v:	ђ6
'assignvariableop_44_adam_dense_8_bias_v:	ђE
6assignvariableop_45_adam_batch_normalization_6_gamma_v:	ђD
5assignvariableop_46_adam_batch_normalization_6_beta_v:	ђ<
)assignvariableop_47_adam_dense_9_kernel_v:	ђ@5
'assignvariableop_48_adam_dense_9_bias_v:@D
6assignvariableop_49_adam_batch_normalization_7_gamma_v:@C
5assignvariableop_50_adam_batch_normalization_7_beta_v:@<
*assignvariableop_51_adam_dense_10_kernel_v:@@6
(assignvariableop_52_adam_dense_10_bias_v:@D
6assignvariableop_53_adam_batch_normalization_8_gamma_v:@C
5assignvariableop_54_adam_batch_normalization_8_beta_v:@<
*assignvariableop_55_adam_dense_11_kernel_v:@6
(assignvariableop_56_adam_dense_11_bias_v:
identity_58ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*э
valueьBЖ:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHт
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*Є
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ├
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*■
_output_shapesв
У::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_6_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_6_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_6_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_6_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_7_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_7_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_7_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_7_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_8_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_8_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_8_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_8_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_11_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_11_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:ј
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_8_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_8_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_6_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_6_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_9_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_9_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_7_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_7_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_10_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_10_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_8_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_8_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_11_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_11_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_8_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_8_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_6_gamma_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_6_beta_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_9_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_9_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_7_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_7_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_10_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_10_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_8_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_8_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_11_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_11_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 х

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: б

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*Є
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
И
З
I__inference_concatenate_2_layer_call_and_return_conditional_losses_784998
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
value	B :м
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10concat/axis:output:0*
N*
T0*'
_output_shapes
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesн
Л:         :         :         :         :         :         :         :         :         :         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10
е
Л
6__inference_batch_normalization_8_layer_call_fn_785285

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ш?
ь

C__inference_model_2_layer_call_and_return_conditional_losses_784545
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
dense_8_784495:	ђ
dense_8_784497:	ђ+
batch_normalization_6_784500:	ђ+
batch_normalization_6_784502:	ђ+
batch_normalization_6_784504:	ђ+
batch_normalization_6_784506:	ђ!
dense_9_784510:	ђ@
dense_9_784512:@*
batch_normalization_7_784515:@*
batch_normalization_7_784517:@*
batch_normalization_7_784519:@*
batch_normalization_7_784521:@!
dense_10_784525:@@
dense_10_784527:@*
batch_normalization_8_784530:@*
batch_normalization_8_784532:@*
batch_normalization_8_784534:@*
batch_normalization_8_784536:@!
dense_11_784539:@
dense_11_784541:
identityѕб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallЩ
concatenate_2/PartitionedCallPartitionedCallfixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_783996Ї
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_784495dense_8_784497*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_784009Є
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_784500batch_normalization_6_784502batch_normalization_6_784504batch_normalization_6_784506*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783381в
dropout_4/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_784029ѕ
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_9_784510dense_9_784512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_784042є
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_784515batch_normalization_7_784517batch_normalization_7_784519batch_normalization_7_784521*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783463Ж
dropout_5/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_784062ї
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_10_784525dense_10_784527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_784075Є
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_8_784530batch_normalization_8_784532batch_normalization_8_784534batch_normalization_8_784536*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783545а
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_11_784539dense_11_784541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_784100x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Я
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Y U
'
_output_shapes
:         
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:         
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:         
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:         
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:         
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:         
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:         
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:         
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:         

_user_specified_nameph_xf:U	Q
'
_output_shapes
:         
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:         
$
_user_specified_name
alcohol_xf
ы
c
*__inference_dropout_5_layer_call_fn_785235

inputs
identityѕбStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_784190o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
џ6
Г
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_785545
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
placeholder_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
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
valueB:Л
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
valueB:█
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
:         ћ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:         *
dtype0	*
shape:         а
PartitionedCallPartitionedCallinputs_alcoholinputs_chloridesplaceholderinputs_densityplaceholder_1placeholder_2	inputs_phPlaceholderWithDefault:output:0placeholder_3inputs_sulphatesplaceholder_4placeholder_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*-
Tin&
$2"	*
Tout
2	*Щ
_output_shapesу
С:         :         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *"
fR
__inference_pruned_782837`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:         b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:         b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:         b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:         b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:         b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:         b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:         c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:         d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*њ
_input_shapesђ
§:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :W S
'
_output_shapes
:         
(
_user_specified_nameinputs/alcohol:YU
'
_output_shapes
:         
*
_user_specified_nameinputs/chlorides:[W
'
_output_shapes
:         
,
_user_specified_nameinputs/citric acid:WS
'
_output_shapes
:         
(
_user_specified_nameinputs/density:]Y
'
_output_shapes
:         
.
_user_specified_nameinputs/fixed acidity:c_
'
_output_shapes
:         
4
_user_specified_nameinputs/free sulfur dioxide:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/pH:^Z
'
_output_shapes
:         
/
_user_specified_nameinputs/residual sugar:YU
'
_output_shapes
:         
*
_user_specified_nameinputs/sulphates:d	`
'
_output_shapes
:         
5
_user_specified_nameinputs/total sulfur dioxide:`
\
'
_output_shapes
:         
1
_user_specified_nameinputs/volatile acidity:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: 
«
Н
6__inference_batch_normalization_6_layer_call_fn_785044

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783428p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
»%
Ь
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783428

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ђ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђѕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ъ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђг
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
О#<Є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђЖ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
и
ѓ
(__inference_model_2_layer_call_fn_784670
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
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ@
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
identityѕбStatefulPartitionedCall╝
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
:         *6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_784107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10
ч	
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_785125

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ђ)
њ
;__inference_transform_features_layer_1_layer_call_fn_785450
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
placeholder_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
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
identity_10к
PartitionedCallPartitionedCallinputs_alcoholinputs_chloridesplaceholderinputs_densityplaceholder_1placeholder_2	inputs_phplaceholder_3inputs_sulphatesplaceholder_4placeholder_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *у
_output_shapesн
Л:         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783712`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:         b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:         b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:         b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:         b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:         b

Identity_7IdentityPartitionedCall:output:7*
T0*'
_output_shapes
:         b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:         b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:         d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*њ
_input_shapesђ
§:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :W S
'
_output_shapes
:         
(
_user_specified_nameinputs/alcohol:YU
'
_output_shapes
:         
*
_user_specified_nameinputs/chlorides:[W
'
_output_shapes
:         
,
_user_specified_nameinputs/citric acid:WS
'
_output_shapes
:         
(
_user_specified_nameinputs/density:]Y
'
_output_shapes
:         
.
_user_specified_nameinputs/fixed acidity:c_
'
_output_shapes
:         
4
_user_specified_nameinputs/free sulfur dioxide:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/pH:^Z
'
_output_shapes
:         
/
_user_specified_nameinputs/residual sugar:YU
'
_output_shapes
:         
*
_user_specified_nameinputs/sulphates:d	`
'
_output_shapes
:         
5
_user_specified_nameinputs/total sulfur dioxide:`
\
'
_output_shapes
:         
1
_user_specified_nameinputs/volatile acidity:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: 
ак
Ѕ
'__inference_serve_tf_examples_fn_783172
examples%
!transform_features_layer_1_783036%
!transform_features_layer_1_783038%
!transform_features_layer_1_783040%
!transform_features_layer_1_783042%
!transform_features_layer_1_783044%
!transform_features_layer_1_783046%
!transform_features_layer_1_783048%
!transform_features_layer_1_783050%
!transform_features_layer_1_783052%
!transform_features_layer_1_783054%
!transform_features_layer_1_783056%
!transform_features_layer_1_783058%
!transform_features_layer_1_783060%
!transform_features_layer_1_783062%
!transform_features_layer_1_783064%
!transform_features_layer_1_783066%
!transform_features_layer_1_783068%
!transform_features_layer_1_783070%
!transform_features_layer_1_783072%
!transform_features_layer_1_783074%
!transform_features_layer_1_783076%
!transform_features_layer_1_783078A
.model_2_dense_8_matmul_readvariableop_resource:	ђ>
/model_2_dense_8_biasadd_readvariableop_resource:	ђN
?model_2_batch_normalization_6_batchnorm_readvariableop_resource:	ђR
Cmodel_2_batch_normalization_6_batchnorm_mul_readvariableop_resource:	ђP
Amodel_2_batch_normalization_6_batchnorm_readvariableop_1_resource:	ђP
Amodel_2_batch_normalization_6_batchnorm_readvariableop_2_resource:	ђA
.model_2_dense_9_matmul_readvariableop_resource:	ђ@=
/model_2_dense_9_biasadd_readvariableop_resource:@M
?model_2_batch_normalization_7_batchnorm_readvariableop_resource:@Q
Cmodel_2_batch_normalization_7_batchnorm_mul_readvariableop_resource:@O
Amodel_2_batch_normalization_7_batchnorm_readvariableop_1_resource:@O
Amodel_2_batch_normalization_7_batchnorm_readvariableop_2_resource:@A
/model_2_dense_10_matmul_readvariableop_resource:@@>
0model_2_dense_10_biasadd_readvariableop_resource:@M
?model_2_batch_normalization_8_batchnorm_readvariableop_resource:@Q
Cmodel_2_batch_normalization_8_batchnorm_mul_readvariableop_resource:@O
Amodel_2_batch_normalization_8_batchnorm_readvariableop_1_resource:@O
Amodel_2_batch_normalization_8_batchnorm_readvariableop_2_resource:@A
/model_2_dense_11_matmul_readvariableop_resource:@>
0model_2_dense_11_biasadd_readvariableop_resource:
identityѕб6model_2/batch_normalization_6/batchnorm/ReadVariableOpб8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1б8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2б:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOpб6model_2/batch_normalization_7/batchnorm/ReadVariableOpб8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1б8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2б:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOpб6model_2/batch_normalization_8/batchnorm/ReadVariableOpб8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1б8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2б:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOpб'model_2/dense_10/BiasAdd/ReadVariableOpб&model_2/dense_10/MatMul/ReadVariableOpб'model_2/dense_11/BiasAdd/ReadVariableOpб&model_2/dense_11/MatMul/ReadVariableOpб&model_2/dense_8/BiasAdd/ReadVariableOpб%model_2/dense_8/MatMul/ReadVariableOpб&model_2/dense_9/BiasAdd/ReadVariableOpб%model_2/dense_9/MatMul/ReadVariableOpU
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
valueB Ё
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*ф
valueаBЮBalcoholB	chloridesBcitric acidBdensityBfixed acidityBfree sulfur dioxideBpHBresidual sugarB	sulphatesBtotal sulfur dioxideBvolatile acidityj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB Ђ
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0ParseExample/Const_10:output:0*
Tdense
2*у
_output_shapesн
Л:         :         :         :         :         :         :         :         :         :         :         *T
dense_shapesD
B:::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 z
 transform_features_layer_1/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(transform_features_layer_1/strided_sliceStridedSlice)transform_features_layer_1/Shape:output:07transform_features_layer_1/strided_slice/stack:output:09transform_features_layer_1/strided_slice/stack_1:output:09transform_features_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
"transform_features_layer_1/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:z
0transform_features_layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2transform_features_layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2transform_features_layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
*transform_features_layer_1/strided_slice_1StridedSlice+transform_features_layer_1/Shape_1:output:09transform_features_layer_1/strided_slice_1/stack:output:0;transform_features_layer_1/strided_slice_1/stack_1:output:0;transform_features_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)transform_features_layer_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :к
'transform_features_layer_1/zeros/packedPack3transform_features_layer_1/strided_slice_1:output:02transform_features_layer_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
&transform_features_layer_1/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R й
 transform_features_layer_1/zerosFill0transform_features_layer_1/zeros/packed:output:0/transform_features_layer_1/zeros/Const:output:0*
T0	*'
_output_shapes
:         ╩
1transform_features_layer_1/PlaceholderWithDefaultPlaceholderWithDefault)transform_features_layer_1/zeros:output:0*'
_output_shapes
:         *
dtype0	*
shape:         Џ
*transform_features_layer_1/PartitionedCallPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:2*ParseExample/ParseExampleV2:dense_values:3*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:6:transform_features_layer_1/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9+ParseExample/ParseExampleV2:dense_values:10!transform_features_layer_1_783036!transform_features_layer_1_783038!transform_features_layer_1_783040!transform_features_layer_1_783042!transform_features_layer_1_783044!transform_features_layer_1_783046!transform_features_layer_1_783048!transform_features_layer_1_783050!transform_features_layer_1_783052!transform_features_layer_1_783054!transform_features_layer_1_783056!transform_features_layer_1_783058!transform_features_layer_1_783060!transform_features_layer_1_783062!transform_features_layer_1_783064!transform_features_layer_1_783066!transform_features_layer_1_783068!transform_features_layer_1_783070!transform_features_layer_1_783072!transform_features_layer_1_783074!transform_features_layer_1_783076!transform_features_layer_1_783078*-
Tin&
$2"	*
Tout
2	*Щ
_output_shapesу
С:         :         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *"
fR
__inference_pruned_782837c
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :п
model_2/concatenate_2/concatConcatV23transform_features_layer_1/PartitionedCall:output:44transform_features_layer_1/PartitionedCall:output:113transform_features_layer_1/PartitionedCall:output:23transform_features_layer_1/PartitionedCall:output:83transform_features_layer_1/PartitionedCall:output:13transform_features_layer_1/PartitionedCall:output:54transform_features_layer_1/PartitionedCall:output:103transform_features_layer_1/PartitionedCall:output:33transform_features_layer_1/PartitionedCall:output:63transform_features_layer_1/PartitionedCall:output:93transform_features_layer_1/PartitionedCall:output:0*model_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ћ
%model_2/dense_8/MatMul/ReadVariableOpReadVariableOp.model_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Е
model_2/dense_8/MatMulMatMul%model_2/concatenate_2/concat:output:0-model_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&model_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
model_2/dense_8/BiasAddBiasAdd model_2/dense_8/MatMul:product:0.model_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђq
model_2/dense_8/ReluRelu model_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ│
6model_2/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0r
-model_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:м
+model_2/batch_normalization_6/batchnorm/addAddV2>model_2/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђЇ
-model_2/batch_normalization_6/batchnorm/RsqrtRsqrt/model_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ╗
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
+model_2/batch_normalization_6/batchnorm/mulMul1model_2/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ╝
-model_2/batch_normalization_6/batchnorm/mul_1Mul"model_2/dense_8/Relu:activations:0/model_2/batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђи
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
-model_2/batch_normalization_6/batchnorm/mul_2Mul@model_2/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђи
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0═
+model_2/batch_normalization_6/batchnorm/subSub@model_2/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ═
-model_2/batch_normalization_6/batchnorm/add_1AddV21model_2/batch_normalization_6/batchnorm/mul_1:z:0/model_2/batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђї
model_2/dropout_4/IdentityIdentity1model_2/batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђЋ
%model_2/dense_9/MatMul/ReadVariableOpReadVariableOp.model_2_dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0д
model_2/dense_9/MatMulMatMul#model_2/dropout_4/Identity:output:0-model_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
&model_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
model_2/dense_9/BiasAddBiasAdd model_2/dense_9/MatMul:product:0.model_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
model_2/dense_9/ReluRelu model_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         @▓
6model_2/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Л
+model_2/batch_normalization_7/batchnorm/addAddV2>model_2/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@ї
-model_2/batch_normalization_7/batchnorm/RsqrtRsqrt/model_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@║
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╬
+model_2/batch_normalization_7/batchnorm/mulMul1model_2/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@╗
-model_2/batch_normalization_7/batchnorm/mul_1Mul"model_2/dense_9/Relu:activations:0/model_2/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @Х
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
-model_2/batch_normalization_7/batchnorm/mul_2Mul@model_2/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@Х
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0╠
+model_2/batch_normalization_7/batchnorm/subSub@model_2/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@╠
-model_2/batch_normalization_7/batchnorm/add_1AddV21model_2/batch_normalization_7/batchnorm/mul_1:z:0/model_2/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @І
model_2/dropout_5/IdentityIdentity1model_2/batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         @ќ
&model_2/dense_10/MatMul/ReadVariableOpReadVariableOp/model_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0е
model_2/dense_10/MatMulMatMul#model_2/dropout_5/Identity:output:0.model_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
'model_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
model_2/dense_10/BiasAddBiasAdd!model_2/dense_10/MatMul:product:0/model_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
model_2/dense_10/ReluRelu!model_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:         @▓
6model_2/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_2/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Л
+model_2/batch_normalization_8/batchnorm/addAddV2>model_2/batch_normalization_8/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:@ї
-model_2/batch_normalization_8/batchnorm/RsqrtRsqrt/model_2/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:@║
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╬
+model_2/batch_normalization_8/batchnorm/mulMul1model_2/batch_normalization_8/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@╝
-model_2/batch_normalization_8/batchnorm/mul_1Mul#model_2/dense_10/Relu:activations:0/model_2/batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @Х
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
-model_2/batch_normalization_8/batchnorm/mul_2Mul@model_2/batch_normalization_8/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:@Х
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0╠
+model_2/batch_normalization_8/batchnorm/subSub@model_2/batch_normalization_8/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@╠
-model_2/batch_normalization_8/batchnorm/add_1AddV21model_2/batch_normalization_8/batchnorm/mul_1:z:0/model_2/batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @ќ
&model_2/dense_11/MatMul/ReadVariableOpReadVariableOp/model_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Х
model_2/dense_11/MatMulMatMul1model_2/batch_normalization_8/batchnorm/add_1:z:0.model_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ћ
'model_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
model_2/dense_11/BiasAddBiasAdd!model_2/dense_11/MatMul:product:0/model_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         p
IdentityIdentity!model_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp7^model_2/batch_normalization_6/batchnorm/ReadVariableOp9^model_2/batch_normalization_6/batchnorm/ReadVariableOp_19^model_2/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_7/batchnorm/ReadVariableOp9^model_2/batch_normalization_7/batchnorm/ReadVariableOp_19^model_2/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_8/batchnorm/ReadVariableOp9^model_2/batch_normalization_8/batchnorm/ReadVariableOp_19^model_2/batch_normalization_8/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp(^model_2/dense_10/BiasAdd/ReadVariableOp'^model_2/dense_10/MatMul/ReadVariableOp(^model_2/dense_11/BiasAdd/ReadVariableOp'^model_2/dense_11/MatMul/ReadVariableOp'^model_2/dense_8/BiasAdd/ReadVariableOp&^model_2/dense_8/MatMul/ReadVariableOp'^model_2/dense_9/BiasAdd/ReadVariableOp&^model_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6model_2/batch_normalization_6/batchnorm/ReadVariableOp6model_2/batch_normalization_6/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_18model_2/batch_normalization_6/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_28model_2/batch_normalization_6/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_7/batchnorm/ReadVariableOp6model_2/batch_normalization_7/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_18model_2/batch_normalization_7/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_28model_2/batch_normalization_7/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_8/batchnorm/ReadVariableOp6model_2/batch_normalization_8/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_18model_2/batch_normalization_8/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_28model_2/batch_normalization_8/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp2R
'model_2/dense_10/BiasAdd/ReadVariableOp'model_2/dense_10/BiasAdd/ReadVariableOp2P
&model_2/dense_10/MatMul/ReadVariableOp&model_2/dense_10/MatMul/ReadVariableOp2R
'model_2/dense_11/BiasAdd/ReadVariableOp'model_2/dense_11/BiasAdd/ReadVariableOp2P
&model_2/dense_11/MatMul/ReadVariableOp&model_2/dense_11/MatMul/ReadVariableOp2P
&model_2/dense_8/BiasAdd/ReadVariableOp&model_2/dense_8/BiasAdd/ReadVariableOp2N
%model_2/dense_8/MatMul/ReadVariableOp%model_2/dense_8/MatMul/ReadVariableOp2P
&model_2/dense_9/BiasAdd/ReadVariableOp&model_2/dense_9/BiasAdd/ReadVariableOp2N
%model_2/dense_9/MatMul/ReadVariableOp%model_2/dense_9/MatMul/ReadVariableOp:M I
#
_output_shapes
:         
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»%
Ь
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785098

inputs6
'assignmovingavg_readvariableop_resource:	ђ8
)assignmovingavg_1_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ0
!batchnorm_readvariableop_resource:	ђ
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ђ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђѕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ъ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ђy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ђг
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
О#<Є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ђ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђЖ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬
░
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783463

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         @z
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╬
░
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785191

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         @z
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я
┤
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783381

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
З@
В

C__inference_model_2_layer_call_and_return_conditional_losses_784383

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
dense_8_784333:	ђ
dense_8_784335:	ђ+
batch_normalization_6_784338:	ђ+
batch_normalization_6_784340:	ђ+
batch_normalization_6_784342:	ђ+
batch_normalization_6_784344:	ђ!
dense_9_784348:	ђ@
dense_9_784350:@*
batch_normalization_7_784353:@*
batch_normalization_7_784355:@*
batch_normalization_7_784357:@*
batch_normalization_7_784359:@!
dense_10_784363:@@
dense_10_784365:@*
batch_normalization_8_784368:@*
batch_normalization_8_784370:@*
batch_normalization_8_784372:@*
batch_normalization_8_784374:@!
dense_11_784377:@
dense_11_784379:
identityѕб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallб!dropout_4/StatefulPartitionedCallб!dropout_5/StatefulPartitionedCall▒
concatenate_2/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_783996Ї
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_784333dense_8_784335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_784009Ё
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_784338batch_normalization_6_784340batch_normalization_6_784342batch_normalization_6_784344*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783428ч
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_784223љ
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_9_784348dense_9_784350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_784042ё
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_784353batch_normalization_7_784355batch_normalization_7_784357batch_normalization_7_784359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783510ъ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_784190ћ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_10_784363dense_10_784365*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_784075Ё
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_8_784368batch_normalization_8_784370batch_normalization_8_784372batch_normalization_8_784374*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783592а
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_11_784377dense_11_784379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_784100x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         е
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O	K
'
_output_shapes
:         
 
_user_specified_nameinputs:O
K
'
_output_shapes
:         
 
_user_specified_nameinputs
љs
м
__inference__traced_save_785761
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop
savev2_const_22

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╬
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*э
valueьBЖ:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*Є
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B в
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopsavev2_const_22"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*ї
_input_shapesЩ
э: :	ђ:ђ:ђ:ђ:ђ:ђ:	ђ@:@:@:@:@:@:@@:@:@:@:@:@:@:: : : : : : : : : :	ђ:ђ:ђ:ђ:	ђ@:@:@:@:@@:@:@:@:@::	ђ:ђ:ђ:ђ:	ђ@:@:@:@:@@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ@: 
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
:	ђ:!

_output_shapes	
:ђ:! 

_output_shapes	
:ђ:!!

_output_shapes	
:ђ:%"!

_output_shapes
:	ђ@: #
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
:	ђ:!-

_output_shapes	
:ђ:!.

_output_shapes	
:ђ:!/

_output_shapes	
:ђ:%0!

_output_shapes
:	ђ@: 1
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
─
Ќ
(__inference_dense_8_layer_call_fn_785007

inputs
unknown:	ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_784009p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▄║
Ј
__inference_pruned_782837

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
	inputs_110
,scale_to_z_score_mean_and_var_identity_input2
.scale_to_z_score_mean_and_var_identity_1_input2
.scale_to_z_score_1_mean_and_var_identity_input4
0scale_to_z_score_1_mean_and_var_identity_1_input2
.scale_to_z_score_2_mean_and_var_identity_input4
0scale_to_z_score_2_mean_and_var_identity_1_input2
.scale_to_z_score_3_mean_and_var_identity_input4
0scale_to_z_score_3_mean_and_var_identity_1_input2
.scale_to_z_score_4_mean_and_var_identity_input4
0scale_to_z_score_4_mean_and_var_identity_1_input2
.scale_to_z_score_5_mean_and_var_identity_input4
0scale_to_z_score_5_mean_and_var_identity_1_input2
.scale_to_z_score_6_mean_and_var_identity_input4
0scale_to_z_score_6_mean_and_var_identity_1_input2
.scale_to_z_score_7_mean_and_var_identity_input4
0scale_to_z_score_7_mean_and_var_identity_1_input2
.scale_to_z_score_8_mean_and_var_identity_input4
0scale_to_z_score_8_mean_and_var_identity_1_input2
.scale_to_z_score_9_mean_and_var_identity_input4
0scale_to_z_score_9_mean_and_var_identity_1_input3
/scale_to_z_score_10_mean_and_var_identity_input5
1scale_to_z_score_10_mean_and_var_identity_1_input
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
identity_11c
scale_to_z_score_10/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_4/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_2/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_7/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    `
scale_to_z_score/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_5/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_8/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_3/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_6/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:         Є
)scale_to_z_score_10/mean_and_var/IdentityIdentity/scale_to_z_score_10_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_10/subSubinputs_copy:output:02scale_to_z_score_10/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         z
scale_to_z_score_10/zeros_like	ZerosLikescale_to_z_score_10/sub:z:0*
T0*'
_output_shapes
:         І
+scale_to_z_score_10/mean_and_var/Identity_1Identity1scale_to_z_score_10_mean_and_var_identity_1_input*
T0*
_output_shapes
: w
scale_to_z_score_10/SqrtSqrt4scale_to_z_score_10/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: љ
scale_to_z_score_10/NotEqualNotEqualscale_to_z_score_10/Sqrt:y:0'scale_to_z_score_10/NotEqual/y:output:0*
T0*
_output_shapes
: r
scale_to_z_score_10/CastCast scale_to_z_score_10/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: ћ
scale_to_z_score_10/addAddV2"scale_to_z_score_10/zeros_like:y:0scale_to_z_score_10/Cast:y:0*
T0*'
_output_shapes
:         ђ
scale_to_z_score_10/Cast_1Castscale_to_z_score_10/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         Њ
scale_to_z_score_10/truedivRealDivscale_to_z_score_10/sub:z:0scale_to_z_score_10/Sqrt:y:0*
T0*'
_output_shapes
:         И
scale_to_z_score_10/SelectV2SelectV2scale_to_z_score_10/Cast_1:y:0scale_to_z_score_10/truediv:z:0scale_to_z_score_10/sub:z:0*
T0*'
_output_shapes
:         m
IdentityIdentity%scale_to_z_score_10/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_4/mean_and_var/IdentityIdentity.scale_to_z_score_4_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_4/subSubinputs_1_copy:output:01scale_to_z_score_4/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_4/zeros_like	ZerosLikescale_to_z_score_4/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_4/mean_and_var/Identity_1Identity0scale_to_z_score_4_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_4/SqrtSqrt3scale_to_z_score_4/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_4/NotEqualNotEqualscale_to_z_score_4/Sqrt:y:0&scale_to_z_score_4/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_4/CastCastscale_to_z_score_4/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_4/addAddV2!scale_to_z_score_4/zeros_like:y:0scale_to_z_score_4/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_4/Cast_1Castscale_to_z_score_4/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_4/truedivRealDivscale_to_z_score_4/sub:z:0scale_to_z_score_4/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_4/SelectV2SelectV2scale_to_z_score_4/Cast_1:y:0scale_to_z_score_4/truediv:z:0scale_to_z_score_4/sub:z:0*
T0*'
_output_shapes
:         n

Identity_1Identity$scale_to_z_score_4/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_2_copyIdentityinputs_2*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_2/mean_and_var/IdentityIdentity.scale_to_z_score_2_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_2/subSubinputs_2_copy:output:01scale_to_z_score_2/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_2/zeros_like	ZerosLikescale_to_z_score_2/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_2/mean_and_var/Identity_1Identity0scale_to_z_score_2_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_2/SqrtSqrt3scale_to_z_score_2/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_2/NotEqualNotEqualscale_to_z_score_2/Sqrt:y:0&scale_to_z_score_2/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_2/CastCastscale_to_z_score_2/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_2/addAddV2!scale_to_z_score_2/zeros_like:y:0scale_to_z_score_2/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_2/Cast_1Castscale_to_z_score_2/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_2/truedivRealDivscale_to_z_score_2/sub:z:0scale_to_z_score_2/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_2/SelectV2SelectV2scale_to_z_score_2/Cast_1:y:0scale_to_z_score_2/truediv:z:0scale_to_z_score_2/sub:z:0*
T0*'
_output_shapes
:         n

Identity_2Identity$scale_to_z_score_2/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_3_copyIdentityinputs_3*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_7/mean_and_var/IdentityIdentity.scale_to_z_score_7_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_7/subSubinputs_3_copy:output:01scale_to_z_score_7/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_7/zeros_like	ZerosLikescale_to_z_score_7/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_7/mean_and_var/Identity_1Identity0scale_to_z_score_7_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_7/SqrtSqrt3scale_to_z_score_7/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_7/NotEqualNotEqualscale_to_z_score_7/Sqrt:y:0&scale_to_z_score_7/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_7/CastCastscale_to_z_score_7/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_7/addAddV2!scale_to_z_score_7/zeros_like:y:0scale_to_z_score_7/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_7/Cast_1Castscale_to_z_score_7/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_7/truedivRealDivscale_to_z_score_7/sub:z:0scale_to_z_score_7/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_7/SelectV2SelectV2scale_to_z_score_7/Cast_1:y:0scale_to_z_score_7/truediv:z:0scale_to_z_score_7/sub:z:0*
T0*'
_output_shapes
:         n

Identity_3Identity$scale_to_z_score_7/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_4_copyIdentityinputs_4*
T0*'
_output_shapes
:         Ђ
&scale_to_z_score/mean_and_var/IdentityIdentity,scale_to_z_score_mean_and_var_identity_input*
T0*
_output_shapes
: ќ
scale_to_z_score/subSubinputs_4_copy:output:0/scale_to_z_score/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         t
scale_to_z_score/zeros_like	ZerosLikescale_to_z_score/sub:z:0*
T0*'
_output_shapes
:         Ё
(scale_to_z_score/mean_and_var/Identity_1Identity.scale_to_z_score_mean_and_var_identity_1_input*
T0*
_output_shapes
: q
scale_to_z_score/SqrtSqrt1scale_to_z_score/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Є
scale_to_z_score/NotEqualNotEqualscale_to_z_score/Sqrt:y:0$scale_to_z_score/NotEqual/y:output:0*
T0*
_output_shapes
: l
scale_to_z_score/CastCastscale_to_z_score/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: І
scale_to_z_score/addAddV2scale_to_z_score/zeros_like:y:0scale_to_z_score/Cast:y:0*
T0*'
_output_shapes
:         z
scale_to_z_score/Cast_1Castscale_to_z_score/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         і
scale_to_z_score/truedivRealDivscale_to_z_score/sub:z:0scale_to_z_score/Sqrt:y:0*
T0*'
_output_shapes
:         г
scale_to_z_score/SelectV2SelectV2scale_to_z_score/Cast_1:y:0scale_to_z_score/truediv:z:0scale_to_z_score/sub:z:0*
T0*'
_output_shapes
:         l

Identity_4Identity"scale_to_z_score/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_5_copyIdentityinputs_5*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_5/mean_and_var/IdentityIdentity.scale_to_z_score_5_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_5/subSubinputs_5_copy:output:01scale_to_z_score_5/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_5/zeros_like	ZerosLikescale_to_z_score_5/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_5/mean_and_var/Identity_1Identity0scale_to_z_score_5_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_5/SqrtSqrt3scale_to_z_score_5/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_5/NotEqualNotEqualscale_to_z_score_5/Sqrt:y:0&scale_to_z_score_5/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_5/CastCastscale_to_z_score_5/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_5/addAddV2!scale_to_z_score_5/zeros_like:y:0scale_to_z_score_5/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_5/Cast_1Castscale_to_z_score_5/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_5/truedivRealDivscale_to_z_score_5/sub:z:0scale_to_z_score_5/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_5/SelectV2SelectV2scale_to_z_score_5/Cast_1:y:0scale_to_z_score_5/truediv:z:0scale_to_z_score_5/sub:z:0*
T0*'
_output_shapes
:         n

Identity_5Identity$scale_to_z_score_5/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_6_copyIdentityinputs_6*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_8/mean_and_var/IdentityIdentity.scale_to_z_score_8_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_8/subSubinputs_6_copy:output:01scale_to_z_score_8/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_8/zeros_like	ZerosLikescale_to_z_score_8/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_8/mean_and_var/Identity_1Identity0scale_to_z_score_8_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_8/SqrtSqrt3scale_to_z_score_8/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_8/NotEqualNotEqualscale_to_z_score_8/Sqrt:y:0&scale_to_z_score_8/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_8/CastCastscale_to_z_score_8/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_8/addAddV2!scale_to_z_score_8/zeros_like:y:0scale_to_z_score_8/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_8/Cast_1Castscale_to_z_score_8/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_8/truedivRealDivscale_to_z_score_8/sub:z:0scale_to_z_score_8/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_8/SelectV2SelectV2scale_to_z_score_8/Cast_1:y:0scale_to_z_score_8/truediv:z:0scale_to_z_score_8/sub:z:0*
T0*'
_output_shapes
:         n

Identity_6Identity$scale_to_z_score_8/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_7_copyIdentityinputs_7*
T0	*'
_output_shapes
:         `

Identity_7Identityinputs_7_copy:output:0*
T0	*'
_output_shapes
:         U
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_3/mean_and_var/IdentityIdentity.scale_to_z_score_3_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_3/subSubinputs_8_copy:output:01scale_to_z_score_3/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_3/zeros_like	ZerosLikescale_to_z_score_3/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_3/mean_and_var/Identity_1Identity0scale_to_z_score_3_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_3/SqrtSqrt3scale_to_z_score_3/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_3/NotEqualNotEqualscale_to_z_score_3/Sqrt:y:0&scale_to_z_score_3/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_3/CastCastscale_to_z_score_3/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_3/addAddV2!scale_to_z_score_3/zeros_like:y:0scale_to_z_score_3/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_3/Cast_1Castscale_to_z_score_3/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_3/truedivRealDivscale_to_z_score_3/sub:z:0scale_to_z_score_3/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_3/SelectV2SelectV2scale_to_z_score_3/Cast_1:y:0scale_to_z_score_3/truediv:z:0scale_to_z_score_3/sub:z:0*
T0*'
_output_shapes
:         n

Identity_8Identity$scale_to_z_score_3/SelectV2:output:0*
T0*'
_output_shapes
:         U
inputs_9_copyIdentityinputs_9*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_9/mean_and_var/IdentityIdentity.scale_to_z_score_9_mean_and_var_identity_input*
T0*
_output_shapes
: џ
scale_to_z_score_9/subSubinputs_9_copy:output:01scale_to_z_score_9/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_9/zeros_like	ZerosLikescale_to_z_score_9/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_9/mean_and_var/Identity_1Identity0scale_to_z_score_9_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_9/SqrtSqrt3scale_to_z_score_9/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_9/NotEqualNotEqualscale_to_z_score_9/Sqrt:y:0&scale_to_z_score_9/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_9/CastCastscale_to_z_score_9/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_9/addAddV2!scale_to_z_score_9/zeros_like:y:0scale_to_z_score_9/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_9/Cast_1Castscale_to_z_score_9/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_9/truedivRealDivscale_to_z_score_9/sub:z:0scale_to_z_score_9/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_9/SelectV2SelectV2scale_to_z_score_9/Cast_1:y:0scale_to_z_score_9/truediv:z:0scale_to_z_score_9/sub:z:0*
T0*'
_output_shapes
:         n

Identity_9Identity$scale_to_z_score_9/SelectV2:output:0*
T0*'
_output_shapes
:         W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_6/mean_and_var/IdentityIdentity.scale_to_z_score_6_mean_and_var_identity_input*
T0*
_output_shapes
: Џ
scale_to_z_score_6/subSubinputs_10_copy:output:01scale_to_z_score_6/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_6/zeros_like	ZerosLikescale_to_z_score_6/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_6/mean_and_var/Identity_1Identity0scale_to_z_score_6_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_6/SqrtSqrt3scale_to_z_score_6/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_6/NotEqualNotEqualscale_to_z_score_6/Sqrt:y:0&scale_to_z_score_6/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_6/CastCastscale_to_z_score_6/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_6/addAddV2!scale_to_z_score_6/zeros_like:y:0scale_to_z_score_6/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_6/Cast_1Castscale_to_z_score_6/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_6/truedivRealDivscale_to_z_score_6/sub:z:0scale_to_z_score_6/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_6/SelectV2SelectV2scale_to_z_score_6/Cast_1:y:0scale_to_z_score_6/truediv:z:0scale_to_z_score_6/sub:z:0*
T0*'
_output_shapes
:         o
Identity_10Identity$scale_to_z_score_6/SelectV2:output:0*
T0*'
_output_shapes
:         W
inputs_11_copyIdentity	inputs_11*
T0*'
_output_shapes
:         Ё
(scale_to_z_score_1/mean_and_var/IdentityIdentity.scale_to_z_score_1_mean_and_var_identity_input*
T0*
_output_shapes
: Џ
scale_to_z_score_1/subSubinputs_11_copy:output:01scale_to_z_score_1/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:         x
scale_to_z_score_1/zeros_like	ZerosLikescale_to_z_score_1/sub:z:0*
T0*'
_output_shapes
:         Ѕ
*scale_to_z_score_1/mean_and_var/Identity_1Identity0scale_to_z_score_1_mean_and_var_identity_1_input*
T0*
_output_shapes
: u
scale_to_z_score_1/SqrtSqrt3scale_to_z_score_1/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: Ї
scale_to_z_score_1/NotEqualNotEqualscale_to_z_score_1/Sqrt:y:0&scale_to_z_score_1/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_1/CastCastscale_to_z_score_1/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Љ
scale_to_z_score_1/addAddV2!scale_to_z_score_1/zeros_like:y:0scale_to_z_score_1/Cast:y:0*
T0*'
_output_shapes
:         ~
scale_to_z_score_1/Cast_1Castscale_to_z_score_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:         љ
scale_to_z_score_1/truedivRealDivscale_to_z_score_1/sub:z:0scale_to_z_score_1/Sqrt:y:0*
T0*'
_output_shapes
:         ┤
scale_to_z_score_1/SelectV2SelectV2scale_to_z_score_1/Cast_1:y:0scale_to_z_score_1/truediv:z:0scale_to_z_score_1/sub:z:0*
T0*'
_output_shapes
:         o
Identity_11Identity$scale_to_z_score_1/SelectV2:output:0*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*Ц
_input_shapesЊ
љ:         :         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :- )
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :-	)
'
_output_shapes
:         :-
)
'
_output_shapes
:         :-)
'
_output_shapes
:         :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 
њ%
Ж
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785225

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:@Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         @h
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
К	
ш
D__inference_dense_11_layer_call_and_return_conditional_losses_784100

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┬
ќ
)__inference_dense_10_layer_call_fn_785261

inputs
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_784075o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я
┤
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785064

inputs0
!batchnorm_readvariableop_resource:	ђ4
%batchnorm_mul_readvariableop_resource:	ђ2
#batchnorm_readvariableop_1_resource:	ђ2
#batchnorm_readvariableop_2_resource:	ђ
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
њ%
Ж
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783510

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:@Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         @h
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ю
Ы
I__inference_concatenate_2_layer_call_and_return_conditional_losses_783996

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
value	B :л
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10concat/axis:output:0*
N*
T0*'
_output_shapes
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesн
Л:         :         :         :         :         :         :         :         :         :         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O	K
'
_output_shapes
:         
 
_user_specified_nameinputs:O
K
'
_output_shapes
:         
 
_user_specified_nameinputs
ї4
з
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783712

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
	inputs_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
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
valueB:Л
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
valueB:█
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
:         ћ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:         *
dtype0	*
shape:         Т
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6PlaceholderWithDefault:output:0inputs_7inputs_8inputs_9	inputs_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*-
Tin&
$2"	*
Tout
2	*Щ
_output_shapesу
С:         :         :         :         :         :         :         :         :         :         :         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *"
fR
__inference_pruned_782837`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:         b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:         b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:         b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:         b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:         b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:         b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:         c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:         d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:         "
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
_construction_contextkEagerRuntime*њ
_input_shapesђ
§:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : : : :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O	K
'
_output_shapes
:         
 
_user_specified_nameinputs:O
K
'
_output_shapes
:         
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: 
Л
Л
$__inference_signature_wrapper_783263
examples
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21:	ђ

unknown_22:	ђ

unknown_23:	ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ

unknown_27:	ђ@

unknown_28:@

unknown_29:@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:
identityѕбStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8ѓ *0
f+R)
'__inference_serve_tf_examples_fn_783172o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:         
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
б

Ш
C__inference_dense_8_layer_call_and_return_conditional_losses_784009

inputs1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є>
ц

C__inference_model_2_layer_call_and_return_conditional_losses_784107

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
dense_8_784010:	ђ
dense_8_784012:	ђ+
batch_normalization_6_784015:	ђ+
batch_normalization_6_784017:	ђ+
batch_normalization_6_784019:	ђ+
batch_normalization_6_784021:	ђ!
dense_9_784043:	ђ@
dense_9_784045:@*
batch_normalization_7_784048:@*
batch_normalization_7_784050:@*
batch_normalization_7_784052:@*
batch_normalization_7_784054:@!
dense_10_784076:@@
dense_10_784078:@*
batch_normalization_8_784081:@*
batch_normalization_8_784083:@*
batch_normalization_8_784085:@*
batch_normalization_8_784087:@!
dense_11_784101:@
dense_11_784103:
identityѕб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб dense_10/StatefulPartitionedCallб dense_11/StatefulPartitionedCallбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCall▒
concatenate_2/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_783996Ї
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_784010dense_8_784012*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_784009Є
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_784015batch_normalization_6_784017batch_normalization_6_784019batch_normalization_6_784021*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_783381в
dropout_4/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_784029ѕ
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_9_784043dense_9_784045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_784042є
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_784048batch_normalization_7_784050batch_normalization_7_784052batch_normalization_7_784054*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783463Ж
dropout_5/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_784062ї
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_10_784076dense_10_784078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_784075Є
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_8_784081batch_normalization_8_784083batch_normalization_8_784085batch_normalization_8_784087*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783545а
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_11_784101dense_11_784103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_784100x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Я
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O	K
'
_output_shapes
:         
 
_user_specified_nameinputs:O
K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЁІ
§
!__inference__wrapped_model_783357
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
.model_2_dense_8_matmul_readvariableop_resource:	ђ>
/model_2_dense_8_biasadd_readvariableop_resource:	ђN
?model_2_batch_normalization_6_batchnorm_readvariableop_resource:	ђR
Cmodel_2_batch_normalization_6_batchnorm_mul_readvariableop_resource:	ђP
Amodel_2_batch_normalization_6_batchnorm_readvariableop_1_resource:	ђP
Amodel_2_batch_normalization_6_batchnorm_readvariableop_2_resource:	ђA
.model_2_dense_9_matmul_readvariableop_resource:	ђ@=
/model_2_dense_9_biasadd_readvariableop_resource:@M
?model_2_batch_normalization_7_batchnorm_readvariableop_resource:@Q
Cmodel_2_batch_normalization_7_batchnorm_mul_readvariableop_resource:@O
Amodel_2_batch_normalization_7_batchnorm_readvariableop_1_resource:@O
Amodel_2_batch_normalization_7_batchnorm_readvariableop_2_resource:@A
/model_2_dense_10_matmul_readvariableop_resource:@@>
0model_2_dense_10_biasadd_readvariableop_resource:@M
?model_2_batch_normalization_8_batchnorm_readvariableop_resource:@Q
Cmodel_2_batch_normalization_8_batchnorm_mul_readvariableop_resource:@O
Amodel_2_batch_normalization_8_batchnorm_readvariableop_1_resource:@O
Amodel_2_batch_normalization_8_batchnorm_readvariableop_2_resource:@A
/model_2_dense_11_matmul_readvariableop_resource:@>
0model_2_dense_11_biasadd_readvariableop_resource:
identityѕб6model_2/batch_normalization_6/batchnorm/ReadVariableOpб8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1б8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2б:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOpб6model_2/batch_normalization_7/batchnorm/ReadVariableOpб8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1б8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2б:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOpб6model_2/batch_normalization_8/batchnorm/ReadVariableOpб8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1б8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2б:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOpб'model_2/dense_10/BiasAdd/ReadVariableOpб&model_2/dense_10/MatMul/ReadVariableOpб'model_2/dense_11/BiasAdd/ReadVariableOpб&model_2/dense_11/MatMul/ReadVariableOpб&model_2/dense_8/BiasAdd/ReadVariableOpб%model_2/dense_8/MatMul/ReadVariableOpб&model_2/dense_9/BiasAdd/ReadVariableOpб%model_2/dense_9/MatMul/ReadVariableOpc
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :┼
model_2/concatenate_2/concatConcatV2fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf*model_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:         Ћ
%model_2/dense_8/MatMul/ReadVariableOpReadVariableOp.model_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Е
model_2/dense_8/MatMulMatMul%model_2/concatenate_2/concat:output:0-model_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&model_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
model_2/dense_8/BiasAddBiasAdd model_2/dense_8/MatMul:product:0.model_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђq
model_2/dense_8/ReluRelu model_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ│
6model_2/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype0r
-model_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:м
+model_2/batch_normalization_6/batchnorm/addAddV2>model_2/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђЇ
-model_2/batch_normalization_6/batchnorm/RsqrtRsqrt/model_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ╗
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
+model_2/batch_normalization_6/batchnorm/mulMul1model_2/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ╝
-model_2/batch_normalization_6/batchnorm/mul_1Mul"model_2/dense_8/Relu:activations:0/model_2/batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђи
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
-model_2/batch_normalization_6/batchnorm/mul_2Mul@model_2/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђи
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђ*
dtype0═
+model_2/batch_normalization_6/batchnorm/subSub@model_2/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ═
-model_2/batch_normalization_6/batchnorm/add_1AddV21model_2/batch_normalization_6/batchnorm/mul_1:z:0/model_2/batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђї
model_2/dropout_4/IdentityIdentity1model_2/batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђЋ
%model_2/dense_9/MatMul/ReadVariableOpReadVariableOp.model_2_dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype0д
model_2/dense_9/MatMulMatMul#model_2/dropout_4/Identity:output:0-model_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
&model_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
model_2/dense_9/BiasAddBiasAdd model_2/dense_9/MatMul:product:0.model_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
model_2/dense_9/ReluRelu model_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         @▓
6model_2/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Л
+model_2/batch_normalization_7/batchnorm/addAddV2>model_2/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:@ї
-model_2/batch_normalization_7/batchnorm/RsqrtRsqrt/model_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:@║
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╬
+model_2/batch_normalization_7/batchnorm/mulMul1model_2/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@╗
-model_2/batch_normalization_7/batchnorm/mul_1Mul"model_2/dense_9/Relu:activations:0/model_2/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @Х
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
-model_2/batch_normalization_7/batchnorm/mul_2Mul@model_2/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:@Х
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0╠
+model_2/batch_normalization_7/batchnorm/subSub@model_2/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@╠
-model_2/batch_normalization_7/batchnorm/add_1AddV21model_2/batch_normalization_7/batchnorm/mul_1:z:0/model_2/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @І
model_2/dropout_5/IdentityIdentity1model_2/batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         @ќ
&model_2/dense_10/MatMul/ReadVariableOpReadVariableOp/model_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0е
model_2/dense_10/MatMulMatMul#model_2/dropout_5/Identity:output:0.model_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ћ
'model_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
model_2/dense_10/BiasAddBiasAdd!model_2/dense_10/MatMul:product:0/model_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
model_2/dense_10/ReluRelu!model_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:         @▓
6model_2/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0r
-model_2/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Л
+model_2/batch_normalization_8/batchnorm/addAddV2>model_2/batch_normalization_8/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:@ї
-model_2/batch_normalization_8/batchnorm/RsqrtRsqrt/model_2/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:@║
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╬
+model_2/batch_normalization_8/batchnorm/mulMul1model_2/batch_normalization_8/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@╝
-model_2/batch_normalization_8/batchnorm/mul_1Mul#model_2/dense_10/Relu:activations:0/model_2/batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @Х
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
-model_2/batch_normalization_8/batchnorm/mul_2Mul@model_2/batch_normalization_8/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:@Х
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0╠
+model_2/batch_normalization_8/batchnorm/subSub@model_2/batch_normalization_8/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@╠
-model_2/batch_normalization_8/batchnorm/add_1AddV21model_2/batch_normalization_8/batchnorm/mul_1:z:0/model_2/batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @ќ
&model_2/dense_11/MatMul/ReadVariableOpReadVariableOp/model_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Х
model_2/dense_11/MatMulMatMul1model_2/batch_normalization_8/batchnorm/add_1:z:0.model_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ћ
'model_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
model_2/dense_11/BiasAddBiasAdd!model_2/dense_11/MatMul:product:0/model_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         p
IdentityIdentity!model_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp7^model_2/batch_normalization_6/batchnorm/ReadVariableOp9^model_2/batch_normalization_6/batchnorm/ReadVariableOp_19^model_2/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_7/batchnorm/ReadVariableOp9^model_2/batch_normalization_7/batchnorm/ReadVariableOp_19^model_2/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_8/batchnorm/ReadVariableOp9^model_2/batch_normalization_8/batchnorm/ReadVariableOp_19^model_2/batch_normalization_8/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp(^model_2/dense_10/BiasAdd/ReadVariableOp'^model_2/dense_10/MatMul/ReadVariableOp(^model_2/dense_11/BiasAdd/ReadVariableOp'^model_2/dense_11/MatMul/ReadVariableOp'^model_2/dense_8/BiasAdd/ReadVariableOp&^model_2/dense_8/MatMul/ReadVariableOp'^model_2/dense_9/BiasAdd/ReadVariableOp&^model_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 2p
6model_2/batch_normalization_6/batchnorm/ReadVariableOp6model_2/batch_normalization_6/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_18model_2/batch_normalization_6/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_28model_2/batch_normalization_6/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_7/batchnorm/ReadVariableOp6model_2/batch_normalization_7/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_18model_2/batch_normalization_7/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_28model_2/batch_normalization_7/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_8/batchnorm/ReadVariableOp6model_2/batch_normalization_8/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_18model_2/batch_normalization_8/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_28model_2/batch_normalization_8/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp2R
'model_2/dense_10/BiasAdd/ReadVariableOp'model_2/dense_10/BiasAdd/ReadVariableOp2P
&model_2/dense_10/MatMul/ReadVariableOp&model_2/dense_10/MatMul/ReadVariableOp2R
'model_2/dense_11/BiasAdd/ReadVariableOp'model_2/dense_11/BiasAdd/ReadVariableOp2P
&model_2/dense_11/MatMul/ReadVariableOp&model_2/dense_11/MatMul/ReadVariableOp2P
&model_2/dense_8/BiasAdd/ReadVariableOp&model_2/dense_8/BiasAdd/ReadVariableOp2N
%model_2/dense_8/MatMul/ReadVariableOp%model_2/dense_8/MatMul/ReadVariableOp2P
&model_2/dense_9/BiasAdd/ReadVariableOp&model_2/dense_9/BiasAdd/ReadVariableOp2N
%model_2/dense_9/MatMul/ReadVariableOp%model_2/dense_9/MatMul/ReadVariableOp:Y U
'
_output_shapes
:         
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:         
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:         
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:         
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:         
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:         
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:         
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:         
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:         

_user_specified_nameph_xf:U	Q
'
_output_shapes
:         
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:         
$
_user_specified_name
alcohol_xf
Џ

ш
D__inference_dense_10_layer_call_and_return_conditional_losses_784075

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┬
ќ
)__inference_dense_11_layer_call_fn_785361

inputs
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_784100o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
д
Л
6__inference_batch_normalization_7_layer_call_fn_785171

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783510o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ш
c
*__inference_dropout_4_layer_call_fn_785108

inputs
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_784223p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ф
┘
.__inference_concatenate_2_layer_call_fn_784982
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
identityЦ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_783996`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesн
Л:         :         :         :         :         :         :         :         :         :         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10
Џ

ш
D__inference_dense_10_layer_call_and_return_conditional_losses_785272

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
њ%
Ж
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785352

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:@Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         @h
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▄
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_784029

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
є
╔
(__inference_model_2_layer_call_fn_784481
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
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ@
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
identityѕбStatefulPartitionedCall§
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
:         *0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_784383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:         
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:         
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:         
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:         
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:         
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:         
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:         
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:         
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:         

_user_specified_nameph_xf:U	Q
'
_output_shapes
:         
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:         
$
_user_specified_name
alcohol_xf
ї
╔
(__inference_model_2_layer_call_fn_784150
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
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ@
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
identityѕбStatefulPartitionedCallЃ
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
:         *6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_784107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:         
*
_user_specified_namefixed_acidity_xf:\X
'
_output_shapes
:         
-
_user_specified_namevolatile_acidity_xf:WS
'
_output_shapes
:         
(
_user_specified_namecitric_acid_xf:ZV
'
_output_shapes
:         
+
_user_specified_nameresidual_sugar_xf:UQ
'
_output_shapes
:         
&
_user_specified_namechlorides_xf:_[
'
_output_shapes
:         
0
_user_specified_namefree_sulfur_dioxide_xf:`\
'
_output_shapes
:         
1
_user_specified_nametotal_sulfur_dioxide_xf:SO
'
_output_shapes
:         
$
_user_specified_name
density_xf:NJ
'
_output_shapes
:         

_user_specified_nameph_xf:U	Q
'
_output_shapes
:         
&
_user_specified_namesulphates_xf:S
O
'
_output_shapes
:         
$
_user_specified_name
alcohol_xf
п
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_784062

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
б

Ш
C__inference_dense_8_layer_call_and_return_conditional_losses_785018

inputs1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
д
Л
6__inference_batch_normalization_8_layer_call_fn_785298

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783592o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╬
░
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_783545

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         @z
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╬
░
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785318

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         @z
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
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
К	
ш
D__inference_dense_11_layer_call_and_return_conditional_losses_785371

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▒
ѓ
(__inference_model_2_layer_call_fn_784725
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
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ@
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
identityѕбStatefulPartitionedCallХ
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
:         *0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_784383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ј
_input_shapesЧ
щ:         :         :         :         :         :         :         :         :         :         :         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10
▄
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_785113

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
е
Л
6__inference_batch_normalization_7_layer_call_fn_785158

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_783463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
п
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_785240

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
├
ќ
(__inference_dense_9_layer_call_fn_785134

inputs
unknown:	ђ@
	unknown_0:@
identityѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_784042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Е
serving_defaultЋ
9
examples-
serving_default_examples:0         <
output_00
StatefulPartitionedCall:0         tensorflow/serving/predict:Рж
м
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
Ц
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
Ж
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
╝
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
╗
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
Ж
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
╝
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator"
_tf_keras_layer
╗
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
Ж
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
╗
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias"
_tf_keras_layer
╦
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
${ _saved_model_loader_tracked_dict"
_tf_keras_model
Х
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
є
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
╦
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
ђlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
П
Ђtrace_0
ѓtrace_1
Ѓtrace_2
ёtrace_32Ж
(__inference_model_2_layer_call_fn_784150
(__inference_model_2_layer_call_fn_784670
(__inference_model_2_layer_call_fn_784725
(__inference_model_2_layer_call_fn_784481┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0zѓtrace_1zЃtrace_2zёtrace_3
╔
Ёtrace_0
єtrace_1
Єtrace_2
ѕtrace_32о
C__inference_model_2_layer_call_and_return_conditional_losses_784818
C__inference_model_2_layer_call_and_return_conditional_losses_784967
C__inference_model_2_layer_call_and_return_conditional_losses_784545
C__inference_model_2_layer_call_and_return_conditional_losses_784609┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0zєtrace_1zЄtrace_2zѕtrace_3
щBШ
!__inference__wrapped_model_783357fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­
	Ѕiter
іbeta_1
Іbeta_2

їdecay
Їlearning_rate,mќ-mЌ5mў6mЎFmџGmЏOmюPmЮ`mъamЪimаjmАsmбtmБ,vц-vЦ5vд6vДFvеGvЕOvфPvФ`vгavГiv«jv»sv░tv▒"
	optimizer
-
јserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Јnon_trainable_variables
љlayers
Љmetrics
 њlayer_regularization_losses
Њlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
З
ћtrace_02Н
.__inference_concatenate_2_layer_call_fn_784982б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zћtrace_0
Ј
Ћtrace_02­
I__inference_concatenate_2_layer_call_and_return_conditional_losses_784998б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЋtrace_0
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
▓
ќnon_trainable_variables
Ќlayers
ўmetrics
 Ўlayer_regularization_losses
џlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ь
Џtrace_02¤
(__inference_dense_8_layer_call_fn_785007б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЏtrace_0
Ѕ
юtrace_02Ж
C__inference_dense_8_layer_call_and_return_conditional_losses_785018б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zюtrace_0
!:	ђ2dense_8/kernel
:ђ2dense_8/bias
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
▓
Юnon_trainable_variables
ъlayers
Ъmetrics
 аlayer_regularization_losses
Аlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
р
бtrace_0
Бtrace_12д
6__inference_batch_normalization_6_layer_call_fn_785031
6__inference_batch_normalization_6_layer_call_fn_785044│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zбtrace_0zБtrace_1
Ќ
цtrace_0
Цtrace_12▄
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785064
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785098│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zцtrace_0zЦtrace_1
 "
trackable_list_wrapper
*:(ђ2batch_normalization_6/gamma
):'ђ2batch_normalization_6/beta
2:0ђ (2!batch_normalization_6/moving_mean
6:4ђ (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
╔
Фtrace_0
гtrace_12ј
*__inference_dropout_4_layer_call_fn_785103
*__inference_dropout_4_layer_call_fn_785108│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zФtrace_0zгtrace_1
 
Гtrace_0
«trace_12─
E__inference_dropout_4_layer_call_and_return_conditional_losses_785113
E__inference_dropout_4_layer_call_and_return_conditional_losses_785125│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zГtrace_0z«trace_1
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
▓
»non_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
Ь
┤trace_02¤
(__inference_dense_9_layer_call_fn_785134б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┤trace_0
Ѕ
хtrace_02Ж
C__inference_dense_9_layer_call_and_return_conditional_losses_785145б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zхtrace_0
!:	ђ@2dense_9/kernel
:@2dense_9/bias
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
▓
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
р
╗trace_0
╝trace_12д
6__inference_batch_normalization_7_layer_call_fn_785158
6__inference_batch_normalization_7_layer_call_fn_785171│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╗trace_0z╝trace_1
Ќ
йtrace_0
Йtrace_12▄
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785191
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785225│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zйtrace_0zЙtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
╔
─trace_0
┼trace_12ј
*__inference_dropout_5_layer_call_fn_785230
*__inference_dropout_5_layer_call_fn_785235│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z─trace_0z┼trace_1
 
кtrace_0
Кtrace_12─
E__inference_dropout_5_layer_call_and_return_conditional_losses_785240
E__inference_dropout_5_layer_call_and_return_conditional_losses_785252│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zкtrace_0zКtrace_1
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
▓
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
№
═trace_02л
)__inference_dense_10_layer_call_fn_785261б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z═trace_0
і
╬trace_02в
D__inference_dense_10_layer_call_and_return_conditional_losses_785272б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╬trace_0
!:@@2dense_10/kernel
:@2dense_10/bias
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
▓
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
р
нtrace_0
Нtrace_12д
6__inference_batch_normalization_8_layer_call_fn_785285
6__inference_batch_normalization_8_layer_call_fn_785298│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zнtrace_0zНtrace_1
Ќ
оtrace_0
Оtrace_12▄
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785318
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785352│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zоtrace_0zОtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
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
▓
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
№
Пtrace_02л
)__inference_dense_11_layer_call_fn_785361б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zПtrace_0
і
яtrace_02в
D__inference_dense_11_layer_call_and_return_conditional_losses_785371б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zяtrace_0
!:@2dense_11/kernel
:2dense_11/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
┌
Сtrace_0
тtrace_12Ъ
;__inference_transform_features_layer_1_layer_call_fn_783779
;__inference_transform_features_layer_1_layer_call_fn_785450б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zСtrace_0zтtrace_1
љ
Тtrace_0
уtrace_12Н
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_785545
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783953б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zТtrace_0zуtrace_1
Ќ
У	_imported
ж_wrapped_function
Ж_structured_inputs
в_structured_outputs
В_output_to_inputs_map"
trackable_dict_wrapper
J
70
81
Q2
R3
k4
l5"
trackable_list_wrapper
к
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
ь0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBц
(__inference_model_2_layer_call_fn_784150fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЯBП
(__inference_model_2_layer_call_fn_784670inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЯBП
(__inference_model_2_layer_call_fn_784725inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ДBц
(__inference_model_2_layer_call_fn_784481fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
C__inference_model_2_layer_call_and_return_conditional_losses_784818inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
C__inference_model_2_layer_call_and_return_conditional_losses_784967inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┬B┐
C__inference_model_2_layer_call_and_return_conditional_losses_784545fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┬B┐
C__inference_model_2_layer_call_and_return_conditional_losses_784609fixed_acidity_xfvolatile_acidity_xfcitric_acid_xfresidual_sugar_xfchlorides_xffree_sulfur_dioxide_xftotal_sulfur_dioxide_xf
density_xfph_xfsulphates_xf
alcohol_xf"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ц
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21B╔
$__inference_signature_wrapper_783263examples"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
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
╔Bк
.__inference_concatenate_2_layer_call_fn_784982inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
СBр
I__inference_concatenate_2_layer_call_and_return_conditional_losses_784998inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_dense_8_layer_call_fn_785007inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_dense_8_layer_call_and_return_conditional_losses_785018inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
чBЭ
6__inference_batch_normalization_6_layer_call_fn_785031inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
6__inference_batch_normalization_6_layer_call_fn_785044inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785064inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785098inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
№BВ
*__inference_dropout_4_layer_call_fn_785103inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
*__inference_dropout_4_layer_call_fn_785108inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
E__inference_dropout_4_layer_call_and_return_conditional_losses_785113inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
E__inference_dropout_4_layer_call_and_return_conditional_losses_785125inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_dense_9_layer_call_fn_785134inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_dense_9_layer_call_and_return_conditional_losses_785145inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
чBЭ
6__inference_batch_normalization_7_layer_call_fn_785158inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
6__inference_batch_normalization_7_layer_call_fn_785171inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785191inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785225inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
№BВ
*__inference_dropout_5_layer_call_fn_785230inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
*__inference_dropout_5_layer_call_fn_785235inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
E__inference_dropout_5_layer_call_and_return_conditional_losses_785240inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
E__inference_dropout_5_layer_call_and_return_conditional_losses_785252inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_dense_10_layer_call_fn_785261inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_dense_10_layer_call_and_return_conditional_losses_785272inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
чBЭ
6__inference_batch_normalization_8_layer_call_fn_785285inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
6__inference_batch_normalization_8_layer_call_fn_785298inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785318inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785352inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_dense_11_layer_call_fn_785361inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_dense_11_layer_call_and_return_conditional_losses_785371inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
н
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21Bщ
;__inference_transform_features_layer_1_layer_call_fn_783779alcohol	chloridescitric aciddensityfixed acidityfree sulfur dioxidepHresidual sugar	sulphatestotal sulfur dioxidevolatile acidity"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
А	
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21Bк
;__inference_transform_features_layer_1_layer_call_fn_785450inputs/alcoholinputs/chloridesinputs/citric acidinputs/densityinputs/fixed acidityinputs/free sulfur dioxide	inputs/pHinputs/residual sugarinputs/sulphatesinputs/total sulfur dioxideinputs/volatile acidity"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
╝	
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21Bр
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_785545inputs/alcoholinputs/chloridesinputs/citric acidinputs/densityinputs/fixed acidityinputs/free sulfur dioxide	inputs/pHinputs/residual sugarinputs/sulphatesinputs/total sulfur dioxideinputs/volatile acidity"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
№
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21Bћ
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783953alcohol	chloridescitric aciddensityfixed acidityfree sulfur dioxidepHresidual sugar	sulphatestotal sulfur dioxidevolatile acidity"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
╚
Ёcreated_variables
є	resources
Єtrackable_objects
ѕinitializers
Ѕassets
і
signatures
$І_self_saveable_object_factories
жtransform_fn"
_generic_user_object
­
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21BЋ
__inference_pruned_782837inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
ї	variables
Ї	keras_api

јtotal

Јcount"
_tf_keras_metric
c
љ	variables
Љ	keras_api

њtotal

Њcount
ћ
_fn_kwargs"
_tf_keras_metric
"J

Const_16jtf.TrackableConstant
"J

Const_15jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
"J

Const_21jtf.TrackableConstant
"J

Const_20jtf.TrackableConstant
"J

Const_19jtf.TrackableConstant
"J

Const_18jtf.TrackableConstant
"J

Const_17jtf.TrackableConstant
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
Ћserving_default"
signature_map
 "
trackable_dict_wrapper
0
ј0
Ј1"
trackable_list_wrapper
.
ї	variables"
_generic_user_object
:  (2total
:  (2count
0
њ0
Њ1"
trackable_list_wrapper
.
љ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
љ
№	capture_0
­	capture_1
ы	capture_2
Ы	capture_3
з	capture_4
З	capture_5
ш	capture_6
Ш	capture_7
э	capture_8
Э	capture_9
щ
capture_10
Щ
capture_11
ч
capture_12
Ч
capture_13
§
capture_14
■
capture_15
 
capture_16
ђ
capture_17
Ђ
capture_18
ѓ
capture_19
Ѓ
capture_20
ё
capture_21Bх
$__inference_signature_wrapper_782897inputsinputs_1	inputs_10	inputs_11inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№	capture_0z­	capture_1zы	capture_2zЫ	capture_3zз	capture_4zЗ	capture_5zш	capture_6zШ	capture_7zэ	capture_8zЭ	capture_9zщ
capture_10zЩ
capture_11zч
capture_12zЧ
capture_13z§
capture_14z■
capture_15z 
capture_16zђ
capture_17zЂ
capture_18zѓ
capture_19zЃ
capture_20zё
capture_21
&:$	ђ2Adam/dense_8/kernel/m
 :ђ2Adam/dense_8/bias/m
/:-ђ2"Adam/batch_normalization_6/gamma/m
.:,ђ2!Adam/batch_normalization_6/beta/m
&:$	ђ@2Adam/dense_9/kernel/m
:@2Adam/dense_9/bias/m
.:,@2"Adam/batch_normalization_7/gamma/m
-:+@2!Adam/batch_normalization_7/beta/m
&:$@@2Adam/dense_10/kernel/m
 :@2Adam/dense_10/bias/m
.:,@2"Adam/batch_normalization_8/gamma/m
-:+@2!Adam/batch_normalization_8/beta/m
&:$@2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
&:$	ђ2Adam/dense_8/kernel/v
 :ђ2Adam/dense_8/bias/v
/:-ђ2"Adam/batch_normalization_6/gamma/v
.:,ђ2!Adam/batch_normalization_6/beta/v
&:$	ђ@2Adam/dense_9/kernel/v
:@2Adam/dense_9/bias/v
.:,@2"Adam/batch_normalization_7/gamma/v
-:+@2!Adam/batch_normalization_7/beta/v
&:$@@2Adam/dense_10/kernel/v
 :@2Adam/dense_10/bias/v
.:,@2"Adam/batch_normalization_8/gamma/v
-:+@2!Adam/batch_normalization_8/beta/v
&:$@2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v▀
!__inference__wrapped_model_783357╣,-8576FGROQP`alikjstвбу
▀б█
пџн
*і'
fixed_acidity_xf         
-і*
volatile_acidity_xf         
(і%
citric_acid_xf         
+і(
residual_sugar_xf         
&і#
chlorides_xf         
0і-
free_sulfur_dioxide_xf         
1і.
total_sulfur_dioxide_xf         
$і!

density_xf         
і
ph_xf         
&і#
sulphates_xf         
$і!

alcohol_xf         
ф "3ф0
.
dense_11"і
dense_11         ╣
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785064d85764б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ╣
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_785098d78564б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ Љ
6__inference_batch_normalization_6_layer_call_fn_785031W85764б1
*б'
!і
inputs         ђ
p 
ф "і         ђЉ
6__inference_batch_normalization_6_layer_call_fn_785044W78564б1
*б'
!і
inputs         ђ
p
ф "і         ђи
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785191bROQP3б0
)б&
 і
inputs         @
p 
ф "%б"
і
0         @
џ и
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_785225bQROP3б0
)б&
 і
inputs         @
p
ф "%б"
і
0         @
џ Ј
6__inference_batch_normalization_7_layer_call_fn_785158UROQP3б0
)б&
 і
inputs         @
p 
ф "і         @Ј
6__inference_batch_normalization_7_layer_call_fn_785171UQROP3б0
)б&
 і
inputs         @
p
ф "і         @и
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785318blikj3б0
)б&
 і
inputs         @
p 
ф "%б"
і
0         @
џ и
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_785352bklij3б0
)б&
 і
inputs         @
p
ф "%б"
і
0         @
џ Ј
6__inference_batch_normalization_8_layer_call_fn_785285Ulikj3б0
)б&
 і
inputs         @
p 
ф "і         @Ј
6__inference_batch_normalization_8_layer_call_fn_785298Uklij3б0
)б&
 і
inputs         @
p
ф "і         @ю
I__inference_concatenate_2_layer_call_and_return_conditional_losses_784998╬цба
ўбћ
ЉџЇ
"і
inputs/0         
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
ф "%б"
і
0         
џ З
.__inference_concatenate_2_layer_call_fn_784982┴цба
ўбћ
ЉџЇ
"і
inputs/0         
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
ф "і         ц
D__inference_dense_10_layer_call_and_return_conditional_losses_785272\`a/б,
%б"
 і
inputs         @
ф "%б"
і
0         @
џ |
)__inference_dense_10_layer_call_fn_785261O`a/б,
%б"
 і
inputs         @
ф "і         @ц
D__inference_dense_11_layer_call_and_return_conditional_losses_785371\st/б,
%б"
 і
inputs         @
ф "%б"
і
0         
џ |
)__inference_dense_11_layer_call_fn_785361Ost/б,
%б"
 і
inputs         @
ф "і         ц
C__inference_dense_8_layer_call_and_return_conditional_losses_785018],-/б,
%б"
 і
inputs         
ф "&б#
і
0         ђ
џ |
(__inference_dense_8_layer_call_fn_785007P,-/б,
%б"
 і
inputs         
ф "і         ђц
C__inference_dense_9_layer_call_and_return_conditional_losses_785145]FG0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         @
џ |
(__inference_dense_9_layer_call_fn_785134PFG0б-
&б#
!і
inputs         ђ
ф "і         @Д
E__inference_dropout_4_layer_call_and_return_conditional_losses_785113^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ Д
E__inference_dropout_4_layer_call_and_return_conditional_losses_785125^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ 
*__inference_dropout_4_layer_call_fn_785103Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ
*__inference_dropout_4_layer_call_fn_785108Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђЦ
E__inference_dropout_5_layer_call_and_return_conditional_losses_785240\3б0
)б&
 і
inputs         @
p 
ф "%б"
і
0         @
џ Ц
E__inference_dropout_5_layer_call_and_return_conditional_losses_785252\3б0
)б&
 і
inputs         @
p
ф "%б"
і
0         @
џ }
*__inference_dropout_5_layer_call_fn_785230O3б0
)б&
 і
inputs         @
p 
ф "і         @}
*__inference_dropout_5_layer_call_fn_785235O3б0
)б&
 і
inputs         @
p
ф "і         @ч
C__inference_model_2_layer_call_and_return_conditional_losses_784545│,-8576FGROQP`alikjstзб№
убс
пџн
*і'
fixed_acidity_xf         
-і*
volatile_acidity_xf         
(і%
citric_acid_xf         
+і(
residual_sugar_xf         
&і#
chlorides_xf         
0і-
free_sulfur_dioxide_xf         
1і.
total_sulfur_dioxide_xf         
$і!

density_xf         
і
ph_xf         
&і#
sulphates_xf         
$і!

alcohol_xf         
p 

 
ф "%б"
і
0         
џ ч
C__inference_model_2_layer_call_and_return_conditional_losses_784609│,-7856FGQROP`aklijstзб№
убс
пџн
*і'
fixed_acidity_xf         
-і*
volatile_acidity_xf         
(і%
citric_acid_xf         
+і(
residual_sugar_xf         
&і#
chlorides_xf         
0і-
free_sulfur_dioxide_xf         
1і.
total_sulfur_dioxide_xf         
$і!

density_xf         
і
ph_xf         
&і#
sulphates_xf         
$і!

alcohol_xf         
p

 
ф "%б"
і
0         
џ ┤
C__inference_model_2_layer_call_and_return_conditional_losses_784818В,-8576FGROQP`alikjstгбе
абю
ЉџЇ
"і
inputs/0         
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
p 

 
ф "%б"
і
0         
џ ┤
C__inference_model_2_layer_call_and_return_conditional_losses_784967В,-7856FGQROP`aklijstгбе
абю
ЉџЇ
"і
inputs/0         
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
p

 
ф "%б"
і
0         
џ М
(__inference_model_2_layer_call_fn_784150д,-8576FGROQP`alikjstзб№
убс
пџн
*і'
fixed_acidity_xf         
-і*
volatile_acidity_xf         
(і%
citric_acid_xf         
+і(
residual_sugar_xf         
&і#
chlorides_xf         
0і-
free_sulfur_dioxide_xf         
1і.
total_sulfur_dioxide_xf         
$і!

density_xf         
і
ph_xf         
&і#
sulphates_xf         
$і!

alcohol_xf         
p 

 
ф "і         М
(__inference_model_2_layer_call_fn_784481д,-7856FGQROP`aklijstзб№
убс
пџн
*і'
fixed_acidity_xf         
-і*
volatile_acidity_xf         
(і%
citric_acid_xf         
+і(
residual_sugar_xf         
&і#
chlorides_xf         
0і-
free_sulfur_dioxide_xf         
1і.
total_sulfur_dioxide_xf         
$і!

density_xf         
і
ph_xf         
&і#
sulphates_xf         
$і!

alcohol_xf         
p

 
ф "і         ї
(__inference_model_2_layer_call_fn_784670▀,-8576FGROQP`alikjstгбе
абю
ЉџЇ
"і
inputs/0         
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
p 

 
ф "і         ї
(__inference_model_2_layer_call_fn_784725▀,-7856FGQROP`aklijstгбе
абю
ЉџЇ
"і
inputs/0         
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
p

 
ф "і         А
__inference_pruned_782837Ѓ,№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃёэбз
вбу
СфЯ
3
alcohol(і%
inputs/alcohol         
7
	chlorides*і'
inputs/chlorides         
;
citric acid,і)
inputs/citric acid         
3
density(і%
inputs/density         
?
fixed acidity.і+
inputs/fixed acidity         
K
free sulfur dioxide4і1
inputs/free sulfur dioxide         
)
pH#і 
	inputs/pH         
3
quality(і%
inputs/quality         	
A
residual sugar/і,
inputs/residual sugar         
7
	sulphates*і'
inputs/sulphates         
M
total sulfur dioxide5і2
inputs/total sulfur dioxide         
E
volatile acidity1і.
inputs/volatile acidity         
ф "пфн
2

alcohol_xf$і!

alcohol_xf         
6
chlorides_xf&і#
chlorides_xf         
:
citric_acid_xf(і%
citric_acid_xf         
2

density_xf$і!

density_xf         
>
fixed_acidity_xf*і'
fixed_acidity_xf         
J
free_sulfur_dioxide_xf0і-
free_sulfur_dioxide_xf         
(
ph_xfі
ph_xf         
2

quality_xf$і!

quality_xf         	
@
residual_sugar_xf+і(
residual_sugar_xf         
6
sulphates_xf&і#
sulphates_xf         
L
total_sulfur_dioxide_xf1і.
total_sulfur_dioxide_xf         
D
volatile_acidity_xf-і*
volatile_acidity_xf         Ё
$__inference_signature_wrapper_782897▄
,№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃёлб╠
б 
─ф└
*
inputs і
inputs         
.
inputs_1"і
inputs_1         
0
	inputs_10#і 
	inputs_10         
0
	inputs_11#і 
	inputs_11         
.
inputs_2"і
inputs_2         
.
inputs_3"і
inputs_3         
.
inputs_4"і
inputs_4         
.
inputs_5"і
inputs_5         
.
inputs_6"і
inputs_6         
.
inputs_7"і
inputs_7         	
.
inputs_8"і
inputs_8         
.
inputs_9"і
inputs_9         "пфн
2

alcohol_xf$і!

alcohol_xf         
6
chlorides_xf&і#
chlorides_xf         
:
citric_acid_xf(і%
citric_acid_xf         
2

density_xf$і!

density_xf         
>
fixed_acidity_xf*і'
fixed_acidity_xf         
J
free_sulfur_dioxide_xf0і-
free_sulfur_dioxide_xf         
(
ph_xfі
ph_xf         
2

quality_xf$і!

quality_xf         	
@
residual_sugar_xf+і(
residual_sugar_xf         
6
sulphates_xf&і#
sulphates_xf         
L
total_sulfur_dioxide_xf1і.
total_sulfur_dioxide_xf         
D
volatile_acidity_xf-і*
volatile_acidity_xf         █
$__inference_signature_wrapper_783263▓@№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃё,-8576FGROQP`alikjst9б6
б 
/ф,
*
examplesі
examples         "3ф0
.
output_0"і
output_0         ╩
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_783953№
,№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃёшбы
жбт
Рфя
,
alcohol!і
alcohol         
0
	chlorides#і 
	chlorides         
4
citric acid%і"
citric acid         
,
density!і
density         
8
fixed acidity'і$
fixed acidity         
D
free sulfur dioxide-і*
free sulfur dioxide         
"
pHі
pH         
:
residual sugar(і%
residual sugar         
0
	sulphates#і 
	sulphates         
F
total sulfur dioxide.і+
total sulfur dioxide         
>
volatile acidity*і'
volatile acidity         
ф "кб┬
║фХ
4

alcohol_xf&і#
0/alcohol_xf         
8
chlorides_xf(і%
0/chlorides_xf         
<
citric_acid_xf*і'
0/citric_acid_xf         
4

density_xf&і#
0/density_xf         
@
fixed_acidity_xf,і)
0/fixed_acidity_xf         
L
free_sulfur_dioxide_xf2і/
0/free_sulfur_dioxide_xf         
*
ph_xf!і
0/ph_xf         
B
residual_sugar_xf-і*
0/residual_sugar_xf         
8
sulphates_xf(і%
0/sulphates_xf         
N
total_sulfur_dioxide_xf3і0
0/total_sulfur_dioxide_xf         
F
volatile_acidity_xf/і,
0/volatile_acidity_xf         
џ Ќ
V__inference_transform_features_layer_1_layer_call_and_return_conditional_losses_785545╝,№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃё┬бЙ
Хб▓
»фФ
3
alcohol(і%
inputs/alcohol         
7
	chlorides*і'
inputs/chlorides         
;
citric acid,і)
inputs/citric acid         
3
density(і%
inputs/density         
?
fixed acidity.і+
inputs/fixed acidity         
K
free sulfur dioxide4і1
inputs/free sulfur dioxide         
)
pH#і 
	inputs/pH         
A
residual sugar/і,
inputs/residual sugar         
7
	sulphates*і'
inputs/sulphates         
M
total sulfur dioxide5і2
inputs/total sulfur dioxide         
E
volatile acidity1і.
inputs/volatile acidity         
ф "кб┬
║фХ
4

alcohol_xf&і#
0/alcohol_xf         
8
chlorides_xf(і%
0/chlorides_xf         
<
citric_acid_xf*і'
0/citric_acid_xf         
4

density_xf&і#
0/density_xf         
@
fixed_acidity_xf,і)
0/fixed_acidity_xf         
L
free_sulfur_dioxide_xf2і/
0/free_sulfur_dioxide_xf         
*
ph_xf!і
0/ph_xf         
B
residual_sugar_xf-і*
0/residual_sugar_xf         
8
sulphates_xf(і%
0/sulphates_xf         
N
total_sulfur_dioxide_xf3і0
0/total_sulfur_dioxide_xf         
F
volatile_acidity_xf/і,
0/volatile_acidity_xf         
џ Ї
;__inference_transform_features_layer_1_layer_call_fn_783779═
,№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃёшбы
жбт
Рфя
,
alcohol!і
alcohol         
0
	chlorides#і 
	chlorides         
4
citric acid%і"
citric acid         
,
density!і
density         
8
fixed acidity'і$
fixed acidity         
D
free sulfur dioxide-і*
free sulfur dioxide         
"
pHі
pH         
:
residual sugar(і%
residual sugar         
0
	sulphates#і 
	sulphates         
F
total sulfur dioxide.і+
total sulfur dioxide         
>
volatile acidity*і'
volatile acidity         
ф "цфа
2

alcohol_xf$і!

alcohol_xf         
6
chlorides_xf&і#
chlorides_xf         
:
citric_acid_xf(і%
citric_acid_xf         
2

density_xf$і!

density_xf         
>
fixed_acidity_xf*і'
fixed_acidity_xf         
J
free_sulfur_dioxide_xf0і-
free_sulfur_dioxide_xf         
(
ph_xfі
ph_xf         
@
residual_sugar_xf+і(
residual_sugar_xf         
6
sulphates_xf&і#
sulphates_xf         
L
total_sulfur_dioxide_xf1і.
total_sulfur_dioxide_xf         
D
volatile_acidity_xf-і*
volatile_acidity_xf         ┌
;__inference_transform_features_layer_1_layer_call_fn_785450џ,№­ыЫзЗшШэЭщЩчЧ§■ ђЂѓЃё┬бЙ
Хб▓
»фФ
3
alcohol(і%
inputs/alcohol         
7
	chlorides*і'
inputs/chlorides         
;
citric acid,і)
inputs/citric acid         
3
density(і%
inputs/density         
?
fixed acidity.і+
inputs/fixed acidity         
K
free sulfur dioxide4і1
inputs/free sulfur dioxide         
)
pH#і 
	inputs/pH         
A
residual sugar/і,
inputs/residual sugar         
7
	sulphates*і'
inputs/sulphates         
M
total sulfur dioxide5і2
inputs/total sulfur dioxide         
E
volatile acidity1і.
inputs/volatile acidity         
ф "цфа
2

alcohol_xf$і!

alcohol_xf         
6
chlorides_xf&і#
chlorides_xf         
:
citric_acid_xf(і%
citric_acid_xf         
2

density_xf$і!

density_xf         
>
fixed_acidity_xf*і'
fixed_acidity_xf         
J
free_sulfur_dioxide_xf0і-
free_sulfur_dioxide_xf         
(
ph_xfі
ph_xf         
@
residual_sugar_xf+і(
residual_sugar_xf         
6
sulphates_xf&і#
sulphates_xf         
L
total_sulfur_dioxide_xf1і.
total_sulfur_dioxide_xf         
D
volatile_acidity_xf-і*
volatile_acidity_xf         