٢
�
�

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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
	separatorstring "serve*2.10.12v2.10.0-76-gfdfc646704c8�m
y
serving_default_inputsPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
|
serving_default_inputs_10Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
|
serving_default_inputs_11Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_3Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_4Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_5Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_6Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_7Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
{
serving_default_inputs_8Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_inputs_9Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
PartitionedCallPartitionedCallserving_default_inputsserving_default_inputs_1serving_default_inputs_10serving_default_inputs_11serving_default_inputs_2serving_default_inputs_3serving_default_inputs_4serving_default_inputs_5serving_default_inputs_6serving_default_inputs_7serving_default_inputs_8serving_default_inputs_9*
Tin
2	*
Tout
2	*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference_signature_wrapper_382

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures* 
* 
* 
* 
* 
* 
* 

serving_default* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
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
GPU 2J 8� *%
f R
__inference__traced_save_427
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
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
GPU 2J 8� *(
f#R!
__inference__traced_restore_437�T
�
E
__inference__traced_restore_437
file_prefix

identity_1��
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
i
__inference__traced_save_427
file_prefix
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
�
�
__inference_pruned_342

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
:���������\
IdentityIdentityinputs_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:���������`

Identity_1Identityinputs_1_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_2_copyIdentityinputs_2*
T0*'
_output_shapes
:���������`

Identity_2Identityinputs_2_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_3_copyIdentityinputs_3*
T0*'
_output_shapes
:���������`

Identity_3Identityinputs_3_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_4_copyIdentityinputs_4*
T0*'
_output_shapes
:���������`

Identity_4Identityinputs_4_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_5_copyIdentityinputs_5*
T0*'
_output_shapes
:���������`

Identity_5Identityinputs_5_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_6_copyIdentityinputs_6*
T0*'
_output_shapes
:���������`

Identity_6Identityinputs_6_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_7_copyIdentityinputs_7*
T0	*'
_output_shapes
:���������`

Identity_7Identityinputs_7_copy:output:0*
T0	*'
_output_shapes
:���������U
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:���������`

Identity_8Identityinputs_8_copy:output:0*
T0*'
_output_shapes
:���������U
inputs_9_copyIdentityinputs_9*
T0*'
_output_shapes
:���������`

Identity_9Identityinputs_9_copy:output:0*
T0*'
_output_shapes
:���������W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:���������b
Identity_10Identityinputs_10_copy:output:0*
T0*'
_output_shapes
:���������W
inputs_11_copyIdentity	inputs_11*
T0*'
_output_shapes
:���������b
Identity_11Identityinputs_11_copy:output:0*
T0*'
_output_shapes
:���������"
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:- )
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-	)
'
_output_shapes
:���������:-
)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������
�
�
!__inference_signature_wrapper_382

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
identity_11�
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2	*
Tout
2	*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *
fR
__inference_pruned_342`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:���������b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:���������b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:���������b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:���������b

Identity_7IdentityPartitionedCall:output:7*
T0	*'
_output_shapes
:���������b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:���������b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:���������d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:���������d
Identity_11IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:���������"
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
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:Q
M
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_9"�	J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
inputs/
serving_default_inputs:0���������
=
inputs_11
serving_default_inputs_1:0���������
?
	inputs_102
serving_default_inputs_10:0���������
?
	inputs_112
serving_default_inputs_11:0���������
=
inputs_21
serving_default_inputs_2:0���������
=
inputs_31
serving_default_inputs_3:0���������
=
inputs_41
serving_default_inputs_4:0���������
=
inputs_51
serving_default_inputs_5:0���������
=
inputs_61
serving_default_inputs_6:0���������
=
inputs_71
serving_default_inputs_7:0	���������
=
inputs_81
serving_default_inputs_8:0���������
=
inputs_91
serving_default_inputs_9:0���������6

alcohol_xf(
PartitionedCall:0���������8
chlorides_xf(
PartitionedCall:1���������:
citric_acid_xf(
PartitionedCall:2���������6

density_xf(
PartitionedCall:3���������<
fixed_acidity_xf(
PartitionedCall:4���������B
free_sulfur_dioxide_xf(
PartitionedCall:5���������1
ph_xf(
PartitionedCall:6���������6

quality_xf(
PartitionedCall:7	���������=
residual_sugar_xf(
PartitionedCall:8���������8
sulphates_xf(
PartitionedCall:9���������D
total_sulfur_dioxide_xf)
PartitionedCall:10���������@
volatile_acidity_xf)
PartitionedCall:11���������tensorflow/serving/predict:�
�
created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures"
_generic_user_object
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
�B�
__inference_pruned_342inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11
,
serving_default"
signature_map
�B�
!__inference_signature_wrapper_382inputsinputs_1	inputs_10	inputs_11inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference_pruned_342����
���
���
3
alcohol(�%
inputs/alcohol���������
7
	chlorides*�'
inputs/chlorides���������
;
citric acid,�)
inputs/citric acid���������
3
density(�%
inputs/density���������
?
fixed acidity.�+
inputs/fixed acidity���������
K
free sulfur dioxide4�1
inputs/free sulfur dioxide���������
)
pH#� 
	inputs/pH���������
3
quality(�%
inputs/quality���������	
A
residual sugar/�,
inputs/residual sugar���������
7
	sulphates*�'
inputs/sulphates���������
M
total sulfur dioxide5�2
inputs/total sulfur dioxide���������
E
volatile acidity1�.
inputs/volatile acidity���������
� "���
2

alcohol_xf$�!

alcohol_xf���������
6
chlorides_xf&�#
chlorides_xf���������
:
citric_acid_xf(�%
citric_acid_xf���������
2

density_xf$�!

density_xf���������
>
fixed_acidity_xf*�'
fixed_acidity_xf���������
J
free_sulfur_dioxide_xf0�-
free_sulfur_dioxide_xf���������
(
ph_xf�
ph_xf���������
2

quality_xf$�!

quality_xf���������	
@
residual_sugar_xf+�(
residual_sugar_xf���������
6
sulphates_xf&�#
sulphates_xf���������
L
total_sulfur_dioxide_xf1�.
total_sulfur_dioxide_xf���������
D
volatile_acidity_xf-�*
volatile_acidity_xf����������

!__inference_signature_wrapper_382�
���
� 
���
*
inputs �
inputs���������
.
inputs_1"�
inputs_1���������
0
	inputs_10#� 
	inputs_10���������
0
	inputs_11#� 
	inputs_11���������
.
inputs_2"�
inputs_2���������
.
inputs_3"�
inputs_3���������
.
inputs_4"�
inputs_4���������
.
inputs_5"�
inputs_5���������
.
inputs_6"�
inputs_6���������
.
inputs_7"�
inputs_7���������	
.
inputs_8"�
inputs_8���������
.
inputs_9"�
inputs_9���������"���
2

alcohol_xf$�!

alcohol_xf���������
6
chlorides_xf&�#
chlorides_xf���������
:
citric_acid_xf(�%
citric_acid_xf���������
2

density_xf$�!

density_xf���������
>
fixed_acidity_xf*�'
fixed_acidity_xf���������
J
free_sulfur_dioxide_xf0�-
free_sulfur_dioxide_xf���������
(
ph_xf�
ph_xf���������
2

quality_xf$�!

quality_xf���������	
@
residual_sugar_xf+�(
residual_sugar_xf���������
6
sulphates_xf&�#
sulphates_xf���������
L
total_sulfur_dioxide_xf1�.
total_sulfur_dioxide_xf���������
D
volatile_acidity_xf-�*
volatile_acidity_xf���������