OPENQASM 2.0;
include "qelib1.inc";
qreg q0[5];
creg c0[5];
u3(0.588920700066562,3.33819085367724,-1.76212674237833) q0[3];
u3(1.83776051771546,2.00694535410684,-0.458968066431490) q0[0];
cx q0[0],q0[3];
u1(2.73658999559140) q0[3];
u3(-2.04922843712856,0.0,0.0) q0[0];
cx q0[3],q0[0];
u3(0.775548741420020,0.0,0.0) q0[0];
cx q0[0],q0[3];
u3(0.688623979445829,-0.732106099973064,0.979844025644105) q0[3];
u3(1.46981726039664,4.07934290664250,-0.658386087005157) q0[0];
u3(2.00705552337345,-1.71156628040743,0.970196074935822) q0[1];
u3(1.72259813595674,-4.09941734351775,0.185783837560155) q0[4];
cx q0[4],q0[1];
u1(2.25297757879679) q0[1];
u3(-2.57534485027807,0.0,0.0) q0[4];
cx q0[1],q0[4];
u3(1.03958908465734,0.0,0.0) q0[4];
cx q0[4],q0[1];
u3(0.506746479654593,2.98134977735330,-0.00533035995814513) q0[1];
u3(0.361757945269060,-4.30657943648764,-0.188231525369836) q0[4];
u3(0.909129381702648,-0.333094995728812,-0.125173982914206) q0[1];
u3(1.34205451441754,-3.26437631671657,0.659898662375451) q0[3];
cx q0[3],q0[1];
u1(2.43015864194230) q0[1];
u3(-1.93257983411912,0.0,0.0) q0[3];
cx q0[1],q0[3];
u3(1.31101708324338,0.0,0.0) q0[3];
cx q0[3],q0[1];
u3(1.07930072394904,3.60734013723459,-2.04461666164925) q0[1];
u3(1.28581786719360,0.591567714296831,5.65455328775451) q0[3];
u3(0.684619214343245,0.571215813497405,0.00134767295752963) q0[0];
u3(0.336370116315106,-2.11652305864660,1.09701029709819) q0[2];
cx q0[2],q0[0];
u1(1.28802652755848) q0[0];
u3(-3.02239049577338,0.0,0.0) q0[2];
cx q0[0],q0[2];
u3(2.10079216633102,0.0,0.0) q0[2];
cx q0[2],q0[0];
u3(1.40285338454177,0.376983274184824,0.0786241501683903) q0[0];
u3(0.879890602258180,4.23158832118056,1.21925585110716) q0[2];
u3(1.00684674351861,1.57908789612537,0.944607526405835) q0[2];
u3(1.35661171352640,0.247292498470403,-3.34375640811956) q0[3];
cx q0[3],q0[2];
u1(3.44681072234772) q0[2];
u3(-0.787878547647407,0.0,0.0) q0[3];
cx q0[2],q0[3];
u3(1.85729455451487,0.0,0.0) q0[3];
cx q0[3],q0[2];
u3(1.09928692616369,0.168921620086982,-1.14280511083772) q0[2];
u3(1.81054841472011,-4.66025148425882,0.784686843460594) q0[3];
u3(1.24112883442033,-2.52929518806878,0.406575653607366) q0[0];
u3(1.82467777945394,-3.88828969067917,0.790818616814684) q0[4];
cx q0[4],q0[0];
u1(1.51291986250798) q0[0];
u3(-0.963595172727066,0.0,0.0) q0[4];
cx q0[0],q0[4];
u3(-0.224836810889549,0.0,0.0) q0[4];
cx q0[4],q0[0];
u3(0.521129756301381,1.45807935437160,-4.09459319373939) q0[0];
u3(0.973784935746631,-4.45467788982079,0.928844463411082) q0[4];
u3(0.670485056104861,0.200735340984409,-2.54879878707041) q0[2];
u3(1.40572923810424,0.832836473812392,-4.88006951561287) q0[0];
cx q0[0],q0[2];
u1(-1.29640731285004) q0[2];
u3(0.461446621888195,0.0,0.0) q0[0];
cx q0[2],q0[0];
u3(3.33606780435026,0.0,0.0) q0[0];
cx q0[0],q0[2];
u3(1.51395550630625,-2.37607535371570,0.885988406243716) q0[2];
u3(2.40183249845990,-1.83788251791847,1.18854920526372) q0[0];
u3(2.71090575122430,0.996207180370309,2.06212427085317) q0[3];
u3(1.31035031892451,-3.17156871711411,-2.59893515831846) q0[4];
cx q0[4],q0[3];
u1(1.40393815408007) q0[3];
u3(-0.466623468188926,0.0,0.0) q0[4];
cx q0[3],q0[4];
u3(3.02760491470022,0.0,0.0) q0[4];
cx q0[4],q0[3];
u3(1.34674119851534,2.32942044996583,-0.892359769496555) q0[3];
u3(2.06403184745486,-0.742423891686717,-1.15216267748647) q0[4];
u3(0.666569196525709,0.826501223161825,-1.45287654878336) q0[1];
u3(0.950201614849266,-0.692065995193061,-1.40931878453631) q0[2];
cx q0[2],q0[1];
u1(3.04504902233039) q0[1];
u3(-2.40693956547035,0.0,0.0) q0[2];
cx q0[1],q0[2];
u3(0.512465761659904,0.0,0.0) q0[2];
cx q0[2],q0[1];
u3(0.955851043160266,-0.704084717062186,3.71333320410002) q0[1];
u3(2.14974475523485,-0.534767336828757,-3.26992976804561) q0[2];
u3(2.09988974826985,-0.231684993208220,-2.04309053543961) q0[3];
u3(1.21598034744330,-3.57156202764191,0.738721114006045) q0[4];
cx q0[4],q0[3];
u1(0.339174526488228) q0[3];
u3(-1.47693486321744,0.0,0.0) q0[4];
cx q0[3],q0[4];
u3(2.39582342622177,0.0,0.0) q0[4];
cx q0[4],q0[3];
u3(1.41856219931710,-0.361906574429995,-1.50270468363841) q0[3];
u3(1.75248598540954,4.35277501166309,1.50618384143837) q0[4];
u3(0.645831697258572,-2.11551028634105,2.34743246988507) q0[2];
u3(0.413879962920943,-3.47411937281416,0.790020465046907) q0[0];
cx q0[0],q0[2];
u1(2.05238506308024) q0[2];
u3(-3.15156965748161,0.0,0.0) q0[0];
cx q0[2],q0[0];
u3(0.913852060047183,0.0,0.0) q0[0];
cx q0[0],q0[2];
u3(0.508342073962054,-0.585913135137041,1.47957152103702) q0[2];
u3(0.435569344507052,-1.16756466414493,-4.57264380957056) q0[0];
u3(2.72675438479985,-1.48151801096768,4.10684649893035) q0[4];
u3(1.35484183125375,0.981718723652164,1.40885408806759) q0[1];
cx q0[1],q0[4];
u1(1.19269522020207) q0[4];
u3(-0.954680629966138,0.0,0.0) q0[1];
cx q0[4],q0[1];
u3(-0.338735387009067,0.0,0.0) q0[1];
cx q0[1],q0[4];
u3(2.38737096474812,0.924768006367241,0.867964868251242) q0[4];
u3(1.54545863241003,1.99251713173657,-0.734929884892456) q0[1];
u3(1.74668515259722,-2.48413598091734,3.06849014473146) q0[1];
u3(0.436280332939606,0.351859374144091,1.44124295111108) q0[4];
cx q0[4],q0[1];
u1(1.66645924935665) q0[1];
u3(-2.14374197118065,0.0,0.0) q0[4];
cx q0[1],q0[4];
u3(0.442162780553103,0.0,0.0) q0[4];
cx q0[4],q0[1];
u3(2.34331992022043,0.510754284303927,-3.32921715747301) q0[1];
u3(1.41062280355611,0.506615728773920,3.05138008489090) q0[4];
u3(2.37798149187784,2.78046064528470,-1.04689579528536) q0[3];
u3(1.61121594471539,0.681340156262155,-1.79759915122540) q0[2];
cx q0[2],q0[3];
u1(1.64669486897062) q0[3];
u3(-0.00355372594057579,0.0,0.0) q0[2];
cx q0[3],q0[2];
u3(2.63894043334539,0.0,0.0) q0[2];
cx q0[2],q0[3];
u3(1.48794087832669,-0.338426732085415,2.48188699455883) q0[3];
u3(0.324509236302935,0.781533241162951,-1.02263276985290) q0[2];
u3(2.07708028356479,0.0406771195273811,1.71844439349919) q0[4];
u3(1.39226301000100,-2.67237848546919,-2.38426149018597) q0[1];
cx q0[1],q0[4];
u1(1.41742846473480) q0[4];
u3(-0.000381534282627216,0.0,0.0) q0[1];
cx q0[4],q0[1];
u3(2.49484869243564,0.0,0.0) q0[1];
cx q0[1],q0[4];
u3(1.06554314223240,1.49090827935944,-0.722869387222459) q0[4];
u3(1.23905737677221,-1.38795331402967,-1.01238457475720) q0[1];
u3(1.33796470895826,0.516979162928382,-1.44035817459716) q0[0];
u3(0.446810609270914,1.25034529882281,-3.39982696971976) q0[2];
cx q0[2],q0[0];
u1(3.18351290555625) q0[0];
u3(-1.73783709989084,0.0,0.0) q0[2];
cx q0[0],q0[2];
u3(0.458403829879995,0.0,0.0) q0[2];
cx q0[2],q0[0];
u3(1.91515558267279,-1.59493916293202,4.35103074348024) q0[0];
u3(2.34320319911149,3.91381036444002,-1.32178639337682) q0[2];
barrier q0[0],q0[1],q0[2],q0[3],q0[4];
measure q0[0] -> c0[0];
measure q0[1] -> c0[1];
measure q0[2] -> c0[2];
measure q0[3] -> c0[3];
measure q0[4] -> c0[4];