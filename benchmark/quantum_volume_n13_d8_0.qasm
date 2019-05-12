OPENQASM 2.0;
include "qelib1.inc";
qreg q0[13];
creg c0[13];
u3(1.24863447995323,1.09549734802925,1.58273837327396) q0[2];
u3(1.95664954338358,-1.64310264723449,-1.37111930035607) q0[8];
cx q0[8],q0[2];
u1(1.73110900030099) q0[2];
u3(-3.00034148467209,0.0,0.0) q0[8];
cx q0[2],q0[8];
u3(1.20384965217679,0.0,0.0) q0[8];
cx q0[8],q0[2];
u3(0.225420254003954,-4.20532547365309,1.88589861704193) q0[2];
u3(1.13153179181579,1.80297135211485,-0.543697493486356) q0[8];
u3(3.07424364168220,2.36540926995765,-1.58499346246949) q0[10];
u3(1.59634627312765,4.86076515831756,-0.249444540234332) q0[11];
cx q0[11],q0[10];
u1(3.60469776815780) q0[10];
u3(-1.44737793528051,0.0,0.0) q0[11];
cx q0[10],q0[11];
u3(2.30898480491760,0.0,0.0) q0[11];
cx q0[11],q0[10];
u3(1.45500819178328,0.133233036506007,-2.30839794228465) q0[10];
u3(1.37026455662199,-0.669742059526629,4.31150589096892) q0[11];
u3(1.89739349294564,2.61328244291576,-0.986988582033064) q0[0];
u3(2.70721234296468,1.66848386286605,-1.61980292733171) q0[12];
cx q0[12],q0[0];
u1(1.48026803120572) q0[0];
u3(-1.01491889076137,0.0,0.0) q0[12];
cx q0[0],q0[12];
u3(-0.424379077325333,0.0,0.0) q0[12];
cx q0[12],q0[0];
u3(2.46465291144330,-3.52174872042800,1.16168884830804) q0[0];
u3(1.00427695819291,-3.83387080758598,-1.39333109131533) q0[12];
u3(1.93508409163936,-2.11181369906238,0.793317369449463) q0[6];
u3(2.22142836424007,-3.80465318173544,-0.124998687426078) q0[3];
cx q0[3],q0[6];
u1(0.181355871107995) q0[6];
u3(-0.890044629981029,0.0,0.0) q0[3];
cx q0[6],q0[3];
u3(1.72478839325345,0.0,0.0) q0[3];
cx q0[3],q0[6];
u3(2.94264216473646,-1.59071903382666,4.55647047700891) q0[6];
u3(2.33901655613638,3.52026296359887,-0.473541279706286) q0[3];
u3(1.93774562074411,0.475724395400421,0.671760931214226) q0[1];
u3(1.99909052050822,-1.22124283395206,-1.32742326455913) q0[5];
cx q0[5],q0[1];
u1(2.14719294819620) q0[1];
u3(-1.74285088376704,0.0,0.0) q0[5];
cx q0[1],q0[5];
u3(3.55015245986158,0.0,0.0) q0[5];
cx q0[5],q0[1];
u3(2.49672971478302,-1.98238255645844,0.545271210287478) q0[1];
u3(1.25934452439476,-1.69977124763360,3.05047394570621) q0[5];
u3(0.372502576707518,1.76598036113944,-1.01446095777689) q0[4];
u3(1.05093077552024,0.592470280497624,-1.67964008366221) q0[7];
cx q0[7],q0[4];
u1(3.25553226558574) q0[4];
u3(-1.17712214976174,0.0,0.0) q0[7];
cx q0[4],q0[7];
u3(2.44782963684521,0.0,0.0) q0[7];
cx q0[7],q0[4];
u3(2.30785807134373,-2.37956665748519,3.31972607252138) q0[4];
u3(1.34822844208303,-0.0286950032866526,5.25263055682837) q0[7];
u3(1.16330694574695,-1.09310590502487,0.622201603261469) q0[3];
u3(0.898301607647296,-2.47615765990968,-0.423279332797221) q0[1];
cx q0[1],q0[3];
u1(1.40584480025428) q0[3];
u3(0.286779902348926,0.0,0.0) q0[1];
cx q0[3],q0[1];
u3(2.06563859671814,0.0,0.0) q0[1];
cx q0[1],q0[3];
u3(1.18808173692672,3.28582013177465,-1.09791497190138) q0[3];
u3(0.987872672486916,-2.80381657841234,0.406344427478206) q0[1];
u3(2.11915124122380,1.59970163603125,0.174604812479738) q0[10];
u3(1.70804365505809,-0.634355583242468,-3.92652169336831) q0[6];
cx q0[6],q0[10];
u1(-0.346831552860275) q0[10];
u3(-2.00535169656716,0.0,0.0) q0[6];
cx q0[10],q0[6];
u3(1.35367143887595,0.0,0.0) q0[6];
cx q0[6],q0[10];
u3(0.896603222187549,2.33689697435173,-3.27120554448293) q0[10];
u3(1.52163109357428,1.63387396064607,-2.89007523492123) q0[6];
u3(1.63515906267968,1.53436354054326,-0.682498024462838) q0[4];
u3(1.90840339481192,-0.248169912180480,-3.57188306702510) q0[2];
cx q0[2],q0[4];
u1(1.95592020486888) q0[4];
u3(-2.16006376637082,0.0,0.0) q0[2];
cx q0[4],q0[2];
u3(1.70830569258126,0.0,0.0) q0[2];
cx q0[2],q0[4];
u3(2.23615807510101,-1.37095103546454,3.45692335238884) q0[4];
u3(1.11650916990493,-1.08802975674127,-1.64612604318235) q0[2];
u3(0.920965599967008,2.63707961817869,-0.844531411854787) q0[7];
u3(0.630554695703744,2.44076080258907,-1.60359024437623) q0[5];
cx q0[5],q0[7];
u1(1.65438582789606) q0[7];
u3(0.268653434945108,0.0,0.0) q0[5];
cx q0[7],q0[5];
u3(0.727411309094631,0.0,0.0) q0[5];
cx q0[5],q0[7];
u3(1.27157370824434,-3.48054074387516,1.21869700798952) q0[7];
u3(0.919580960001008,-3.49843389226429,0.870254984519677) q0[5];
u3(1.56359748612482,1.29215797991955,0.491467772652100) q0[9];
u3(0.0951866941953302,-2.03544036730740,-2.26894287417241) q0[8];
cx q0[8],q0[9];
u1(2.95568136771108) q0[9];
u3(-2.59667884747598,0.0,0.0) q0[8];
cx q0[9],q0[8];
u3(1.42928694483673,0.0,0.0) q0[8];
cx q0[8],q0[9];
u3(1.92240791383009,-1.52552571697874,-0.738065425785975) q0[9];
u3(0.757972862418020,1.13610932688342,-4.65373907351455) q0[8];
u3(0.908677045260954,-1.01769447626528,1.72778323133706) q0[0];
u3(0.634090651496744,2.26637299524362,-3.63168701240400) q0[11];
cx q0[11],q0[0];
u1(1.60337751321253) q0[0];
u3(-3.16577672242221,0.0,0.0) q0[11];
cx q0[0],q0[11];
u3(2.48652327347576,0.0,0.0) q0[11];
cx q0[11],q0[0];
u3(0.772097737576080,-0.0920263348898679,1.16376196222349) q0[0];
u3(1.83748182180120,-4.97282183324173,-1.14409114844050) q0[11];
u3(1.55274029278917,-1.17857050098499,-1.27227212070495) q0[5];
u3(1.02395533290419,-4.60745822144027,0.621134637556030) q0[6];
cx q0[6],q0[5];
u1(1.05962833832296) q0[5];
u3(-0.536831260606293,0.0,0.0) q0[6];
cx q0[5],q0[6];
u3(3.22522987898990,0.0,0.0) q0[6];
cx q0[6],q0[5];
u3(0.685550274159522,0.329684894094724,-3.01746138424534) q0[5];
u3(1.24964484906975,-4.98840265700323,0.461233060883958) q0[6];
u3(0.454949761243443,0.219196759875725,0.377542327673246) q0[3];
u3(1.22471298643941,-0.377366188608923,-2.32257733381795) q0[1];
cx q0[1],q0[3];
u1(1.25055017642405) q0[3];
u3(-0.726610866123213,0.0,0.0) q0[1];
cx q0[3],q0[1];
u3(-0.392342286188990,0.0,0.0) q0[1];
cx q0[1],q0[3];
u3(0.472686629624139,0.955528966139675,-0.337020471378773) q0[3];
u3(2.75331819525436,3.43170542271926,1.14504641969039) q0[1];
u3(0.582732217555227,4.22147499357165,-1.70949238830891) q0[0];
u3(1.02788997713360,1.49090798827654,-1.20148886600047) q0[11];
cx q0[11],q0[0];
u1(-0.0995898193819249) q0[0];
u3(-2.37447633831881,0.0,0.0) q0[11];
cx q0[0],q0[11];
u3(1.03629490926770,0.0,0.0) q0[11];
cx q0[11],q0[0];
u3(1.92679472474596,-0.702428750549787,-1.42965623420072) q0[0];
u3(1.00376218994600,-3.13781206868819,1.03232692536509) q0[11];
u3(0.841097604863622,-2.34544695055027,0.906338062507541) q0[9];
u3(1.50723828625278,-3.97595258573689,-0.175806698300190) q0[7];
cx q0[7],q0[9];
u1(1.17872972829798) q0[9];
u3(0.0699757456235086,0.0,0.0) q0[7];
cx q0[9],q0[7];
u3(2.00538336237634,0.0,0.0) q0[7];
cx q0[7],q0[9];
u3(1.81084554281709,-1.16429018548941,-1.19907874413079) q0[9];
u3(2.46686424402521,-0.993511957831377,1.43244365930215) q0[7];
u3(2.15616174299340,1.58153036203563,-3.60741759986684) q0[4];
u3(0.971338611851363,2.80688611545734,-2.57333874911357) q0[10];
cx q0[10],q0[4];
u1(0.412961056627683) q0[4];
u3(-1.18404910569266,0.0,0.0) q0[10];
cx q0[4],q0[10];
u3(2.62154685625490,0.0,0.0) q0[10];
cx q0[10],q0[4];
u3(1.65413422590740,0.537162175644102,-2.38455963728822) q0[4];
u3(0.659509313344660,5.13460392392688,0.136070350257996) q0[10];
u3(1.24783020033684,-0.385845535267999,-1.15591892159279) q0[12];
u3(1.40392188753869,-3.38821496555419,0.0359222342770078) q0[2];
cx q0[2],q0[12];
u1(-0.248736373113011) q0[12];
u3(-1.92553440684608,0.0,0.0) q0[2];
cx q0[12],q0[2];
u3(0.969766820113447,0.0,0.0) q0[2];
cx q0[2],q0[12];
u3(0.900701939540880,-1.56646468613900,-0.305750179000498) q0[12];
u3(0.746498126788651,3.33856565119055,2.87797358579900) q0[2];
u3(1.06011589437183,1.30040929638717,-3.28533572273857) q0[4];
u3(1.73189735573841,3.04612787071016,-2.96867613335227) q0[0];
cx q0[0],q0[4];
u1(2.61695601748407) q0[4];
u3(0.293217841691730,0.0,0.0) q0[0];
cx q0[4],q0[0];
u3(1.53795138149160,0.0,0.0) q0[0];
cx q0[0],q0[4];
u3(1.22615461788720,1.90536088085620,-3.98409304250051) q0[4];
u3(1.57320598914287,0.223148794838024,5.64621709071983) q0[0];
u3(2.59240001228933,-0.256733183970861,2.53092584840186) q0[9];
u3(2.10807935815035,-1.33398539406862,-1.32006020923177) q0[6];
cx q0[6],q0[9];
u1(1.73829972330624) q0[9];
u3(-2.07600902881664,0.0,0.0) q0[6];
cx q0[9],q0[6];
u3(3.47569952669331,0.0,0.0) q0[6];
cx q0[6],q0[9];
u3(1.78191385224749,-0.288876190573809,-1.36124857306058) q0[9];
u3(2.52004161359989,4.24630081419702,-1.89195879246437) q0[6];
u3(2.55765266340511,0.575710137560289,1.73142355290747) q0[5];
u3(1.41355936605421,-3.29568622827449,-2.66672855380602) q0[1];
cx q0[1],q0[5];
u1(0.0268145083806417) q0[5];
u3(-0.750230235144044,0.0,0.0) q0[1];
cx q0[5],q0[1];
u3(1.66787990886332,0.0,0.0) q0[1];
cx q0[1],q0[5];
u3(2.56147634768205,-0.522052168445843,1.44437634931529) q0[5];
u3(2.37555641248923,2.35447922207258,-3.48241671150682) q0[1];
u3(2.25508429073763,2.94479132674378,-1.33588850502358) q0[12];
u3(2.20878327109395,1.13122816170788,-2.99698477708753) q0[7];
cx q0[7],q0[12];
u1(2.28144325815543) q0[12];
u3(-2.04649515204578,0.0,0.0) q0[7];
cx q0[12],q0[7];
u3(0.0110181757094461,0.0,0.0) q0[7];
cx q0[7],q0[12];
u3(1.37207214531731,1.64638639986712,-2.47152061407868) q0[12];
u3(0.895673018598515,-3.14954469271859,-3.00209244809488) q0[7];
u3(1.82865367414042,0.695604010379994,-3.78821857907605) q0[3];
u3(0.941055569073473,2.88200952084973,-3.09426553674637) q0[2];
cx q0[2],q0[3];
u1(2.99240317458144) q0[3];
u3(-1.90197339664230,0.0,0.0) q0[2];
cx q0[3],q0[2];
u3(1.58651423756153,0.0,0.0) q0[2];
cx q0[2],q0[3];
u3(2.61355700904985,0.537585077467429,-1.54117535835954) q0[3];
u3(2.01135223309033,-0.111156223124026,-1.54486971962679) q0[2];
u3(2.67527867874714,0.946972282105988,-2.70315232432046) q0[11];
u3(2.56569082226897,3.82937493719461,-1.62159551965210) q0[10];
cx q0[10],q0[11];
u1(2.71328109631474) q0[11];
u3(-1.57746208631313,0.0,0.0) q0[10];
cx q0[11],q0[10];
u3(0.181461110016597,0.0,0.0) q0[10];
cx q0[10],q0[11];
u3(2.53952403924091,-4.07590729025790,1.14423488108002) q0[11];
u3(1.60137483516448,-2.72621890827393,-1.80940081311216) q0[10];
u3(1.23238271225193,0.847846924684939,1.24721211333671) q0[6];
u3(0.927681952275311,-0.285736642256466,-3.16571703910262) q0[12];
cx q0[12],q0[6];
u1(1.41006883089873) q0[6];
u3(-3.27881346618071,0.0,0.0) q0[12];
cx q0[6],q0[12];
u3(2.66605805501928,0.0,0.0) q0[12];
cx q0[12],q0[6];
u3(1.13319512538476,0.0442600249781285,-1.90397512648822) q0[6];
u3(0.970889311339326,-1.38709404806985,2.83943486472910) q0[12];
u3(2.24459570365507,1.31722906512862,-3.96581234615977) q0[1];
u3(1.76304280107039,2.82682083745039,-2.54205757703952) q0[11];
cx q0[11],q0[1];
u1(0.267802040310713) q0[1];
u3(0.985963803597735,0.0,0.0) q0[11];
cx q0[1],q0[11];
u3(3.14331603774115,0.0,0.0) q0[11];
cx q0[11],q0[1];
u3(2.39823727264684,-1.44093575848015,-0.952806967162381) q0[1];
u3(1.20577425756368,-1.08988623863297,-3.16202875718096) q0[11];
u3(1.78017422620891,1.34705945272048,-2.46456538971878) q0[8];
u3(1.23942140547553,2.21375231538469,-3.52483639449506) q0[2];
cx q0[2],q0[8];
u1(3.55190302897812) q0[8];
u3(-0.885946001921486,0.0,0.0) q0[2];
cx q0[8],q0[2];
u3(1.53276541836600,0.0,0.0) q0[2];
cx q0[2],q0[8];
u3(1.52320450920053,-0.0960985816753852,3.20330842168865) q0[8];
u3(3.07650346000460,-4.08906536971927,0.150980112374096) q0[2];
u3(1.83775628146153,1.43684880905594,-3.95497957027237) q0[9];
u3(2.21084132897447,-2.57052641493905,3.32168170872546) q0[5];
cx q0[5],q0[9];
u1(2.21069827899886) q0[9];
u3(0.176984811133479,0.0,0.0) q0[5];
cx q0[9],q0[5];
u3(1.04417505205108,0.0,0.0) q0[5];
cx q0[5],q0[9];
u3(2.26476537860202,-0.507990344287802,3.57636780824779) q0[9];
u3(2.71967556204557,0.718736509144910,2.51100198756812) q0[5];
u3(0.704642740435143,2.23189101274793,-1.16076742764963) q0[10];
u3(0.304010587985815,0.532755832744760,-1.65534905385846) q0[0];
cx q0[0],q0[10];
u1(1.94164866267498) q0[10];
u3(-1.78442195391972,0.0,0.0) q0[0];
cx q0[10],q0[0];
u3(2.82149183625384,0.0,0.0) q0[0];
cx q0[0],q0[10];
u3(0.0447590707176649,-1.43714206554228,2.32258836504238) q0[10];
u3(0.285515957354191,4.86745210913404,-0.370499088362647) q0[0];
u3(1.13590337640265,-1.53184959928970,-0.511596802273268) q0[7];
u3(1.92559384132533,-1.80923977911094,0.715834704824952) q0[4];
cx q0[4],q0[7];
u1(1.01312237760599) q0[7];
u3(-1.61698068036119,0.0,0.0) q0[4];
cx q0[7],q0[4];
u3(2.36643093488175,0.0,0.0) q0[4];
cx q0[4],q0[7];
u3(2.58862184282387,2.37332222101087,-1.91833097932324) q0[7];
u3(1.82697614197814,2.03744558347512,1.93872516204035) q0[4];
u3(1.07916724918180,-0.0312578275422802,1.50057587149593) q0[8];
u3(1.69267628058492,-0.124130565992815,-2.05810313691866) q0[10];
cx q0[10],q0[8];
u1(-0.502612577607521) q0[8];
u3(1.07188194550062,0.0,0.0) q0[10];
cx q0[8],q0[10];
u3(3.17371806304340,0.0,0.0) q0[10];
cx q0[10],q0[8];
u3(2.03338953167046,-0.370192317089256,2.94093086393475) q0[8];
u3(0.964030819199067,1.85589504080275,2.98803270459964) q0[10];
u3(2.01593773908348,3.34867834840955,-0.578690718545002) q0[6];
u3(2.04659787183528,1.84644341234867,-1.01984106260674) q0[1];
cx q0[1],q0[6];
u1(3.11378618572205) q0[6];
u3(-2.40091757579896,0.0,0.0) q0[1];
cx q0[6],q0[1];
u3(1.02182271349471,0.0,0.0) q0[1];
cx q0[1],q0[6];
u3(1.65858908106104,-4.26225663065026,1.36706308672735) q0[6];
u3(1.65759572918627,-1.68726877231257,-1.08175944607147) q0[1];
u3(1.65390781883173,3.21710852264660,-1.11165284671593) q0[2];
u3(0.584690844994268,1.56654940801513,-1.04000733697093) q0[3];
cx q0[3],q0[2];
u1(1.77034204500401) q0[2];
u3(-2.79440494930176,0.0,0.0) q0[3];
cx q0[2],q0[3];
u3(0.0366601837366329,0.0,0.0) q0[3];
cx q0[3],q0[2];
u3(2.51072841496445,-2.38638677147589,-0.419717989817583) q0[2];
u3(1.41396030143799,-1.56589155945187,-0.745190469785098) q0[3];
u3(1.91143435760614,-0.887473139715649,2.41244262137395) q0[12];
u3(1.45048133656818,-2.10911759751007,-1.82125198147790) q0[7];
cx q0[7],q0[12];
u1(2.50299347938827) q0[12];
u3(-1.39090525376563,0.0,0.0) q0[7];
cx q0[12],q0[7];
u3(0.0234701170404634,0.0,0.0) q0[7];
cx q0[7],q0[12];
u3(1.06307652218843,1.52660286821481,-2.57725302565903) q0[12];
u3(0.229911053545531,-0.368127602026838,-4.81168793631718) q0[7];
u3(0.979318456297499,-2.62442938694085,1.51250767070405) q0[5];
u3(0.538855569118117,-2.55880243404978,0.941792425329291) q0[11];
cx q0[11],q0[5];
u1(3.62725482223801) q0[5];
u3(-0.974441044974481,0.0,0.0) q0[11];
cx q0[5],q0[11];
u3(1.94075250551091,0.0,0.0) q0[11];
cx q0[11],q0[5];
u3(2.27188710525945,-0.834757844038662,0.148176922544811) q0[5];
u3(1.32002083736627,0.688447037445828,5.40932530356270) q0[11];
u3(1.43904523633095,1.83689775326244,-2.49627897073835) q0[4];
u3(1.62254560185975,-2.14416951572280,2.91482788330853) q0[0];
cx q0[0],q0[4];
u1(0.720693199585855) q0[4];
u3(-3.07416630960251,0.0,0.0) q0[0];
cx q0[4],q0[0];
u3(2.17639709091008,0.0,0.0) q0[0];
cx q0[0],q0[4];
u3(2.40984201196555,-1.89061525390436,1.60612551594995) q0[4];
u3(1.52059596794966,0.688978240653307,4.37062759115798) q0[0];
u3(1.95731992131840,2.67142782483856,-1.50444788416287) q0[11];
u3(1.92383633814864,2.28239391333152,-0.655397866819855) q0[5];
cx q0[5],q0[11];
u1(1.62136529067188) q0[11];
u3(-2.51698851346054,0.0,0.0) q0[5];
cx q0[11],q0[5];
u3(0.156513410124795,0.0,0.0) q0[5];
cx q0[5],q0[11];
u3(1.99437563777097,-2.39968848181153,1.54693014126101) q0[11];
u3(1.17076639881689,3.19448951453017,1.63590141277643) q0[5];
u3(0.717957054232321,1.45516172049699,-1.79032132064885) q0[10];
u3(0.496732051448584,-3.76212185854028,1.48574939220605) q0[0];
cx q0[0],q0[10];
u1(0.607590144965633) q0[10];
u3(1.26330264884245,0.0,0.0) q0[0];
cx q0[10],q0[0];
u3(2.83400359274895,0.0,0.0) q0[0];
cx q0[0],q0[10];
u3(0.921658043550649,0.983465312490652,-1.33296625348413) q0[10];
u3(0.200627416850818,1.49539488083395,3.14289562728784) q0[0];
u3(2.07755302542199,-1.39086748453716,-0.774845896598188) q0[9];
u3(1.13873115531228,-4.52673297104005,1.17662364406785) q0[1];
cx q0[1],q0[9];
u1(1.16650843078706) q0[9];
u3(-3.74414549331147,0.0,0.0) q0[1];
cx q0[9],q0[1];
u3(1.88023454038586,0.0,0.0) q0[1];
cx q0[1],q0[9];
u3(1.20502377206936,2.95632970716166,-1.12633517359514) q0[9];
u3(1.54484273224129,-2.44211472447260,-2.38199006896626) q0[1];
u3(1.66093078161327,-0.368766706729608,0.0161850693346587) q0[12];
u3(1.95544677938792,-2.85976781403820,0.835191716430273) q0[4];
cx q0[4],q0[12];
u1(1.10530304510082) q0[12];
u3(-1.35680354335507,0.0,0.0) q0[4];
cx q0[12],q0[4];
u3(2.41840303719764,0.0,0.0) q0[4];
cx q0[4],q0[12];
u3(1.55306366195488,1.81256914554387,-2.83527012867182) q0[12];
u3(1.65461154522692,2.19965359563665,3.53042064475127) q0[4];
u3(1.72168214425418,1.49969273645507,-2.09782429809319) q0[8];
u3(1.54059950827630,-2.28734640120228,2.70313565413291) q0[2];
cx q0[2],q0[8];
u1(0.412388122985231) q0[8];
u3(-1.63244432512185,0.0,0.0) q0[2];
cx q0[8],q0[2];
u3(2.59608692743720,0.0,0.0) q0[2];
cx q0[2],q0[8];
u3(2.35642860826813,-1.97126801429767,2.10971797639265) q0[8];
u3(1.60700359643819,0.536414349983503,1.85235745123513) q0[2];
u3(1.16522550841570,-0.885858332071200,1.09385688807709) q0[7];
u3(1.82407606290335,-1.30133151583553,-1.89501816099163) q0[3];
cx q0[3],q0[7];
u1(-0.0571009579698769) q0[7];
u3(-1.56806110430397,0.0,0.0) q0[3];
cx q0[7],q0[3];
u3(0.685092229821506,0.0,0.0) q0[3];
cx q0[3],q0[7];
u3(1.86025315601897,0.595243658312205,0.254218596264425) q0[7];
u3(2.12057486189374,-0.113707951634697,-2.82655831230258) q0[3];
u3(1.54448440106690,0.958544425243346,-3.93193414325591) q0[5];
u3(2.42337819054789,3.11263267694796,-2.13655235755658) q0[2];
cx q0[2],q0[5];
u1(1.59086579620462) q0[5];
u3(-0.236276104732108,0.0,0.0) q0[2];
cx q0[5],q0[2];
u3(2.10834863470249,0.0,0.0) q0[2];
cx q0[2],q0[5];
u3(1.44137341415109,1.36469663645566,-1.33206344026872) q0[5];
u3(0.982614096719755,0.382644946512265,-0.758311637052967) q0[2];
u3(0.829766996677897,-0.0981631037979646,-1.95197307835912) q0[1];
u3(0.964360483428953,-4.95299961834464,0.763739646327503) q0[12];
cx q0[12],q0[1];
u1(0.998171487678339) q0[1];
u3(-2.91134941316572,0.0,0.0) q0[12];
cx q0[1],q0[12];
u3(1.94399241532217,0.0,0.0) q0[12];
cx q0[12],q0[1];
u3(1.33791270175182,-1.06477847646281,-1.02837881343125) q0[1];
u3(0.266066642935363,-0.440673782484166,-3.62187017024886) q0[12];
u3(0.438707466511355,-4.17209968226102,1.85094417444049) q0[9];
u3(1.55904644087859,2.67379957066473,-2.86319128826850) q0[0];
cx q0[0],q0[9];
u1(-0.224624583402441) q0[9];
u3(-1.75458779381523,0.0,0.0) q0[0];
cx q0[9],q0[0];
u3(0.562875490469687,0.0,0.0) q0[0];
cx q0[0],q0[9];
u3(2.90586792899834,-0.550796623595072,2.12630760458384) q0[9];
u3(1.27411178925858,-4.85541952911493,-0.253337129016769) q0[0];
u3(1.18879796910255,1.18153854752359,-1.92789164668728) q0[6];
u3(2.17889394345734,-4.00006893628982,2.16355600813308) q0[4];
cx q0[4],q0[6];
u1(1.84414036379784) q0[6];
u3(-0.0525686108432586,0.0,0.0) q0[4];
cx q0[6],q0[4];
u3(0.712810910828384,0.0,0.0) q0[4];
cx q0[4],q0[6];
u3(1.55222115513101,-1.12933409450363,2.07216056272114) q0[6];
u3(1.69712554925648,-4.98966355477089,0.603677320382933) q0[4];
u3(1.14025924400780,-0.257072549313602,-1.60813525580174) q0[7];
u3(1.55539744256244,-4.95620828348195,0.729301435675239) q0[11];
cx q0[11],q0[7];
u1(1.73886040390169) q0[7];
u3(-2.96866712879527,0.0,0.0) q0[11];
cx q0[7],q0[11];
u3(1.30700263253138,0.0,0.0) q0[11];
cx q0[11],q0[7];
u3(0.813739486856285,-1.11324389368069,4.10827723289924) q0[7];
u3(1.82665359113348,-3.20454410128240,1.89393565800013) q0[11];
u3(1.62488669626364,-1.76600150334756,1.31523384409223) q0[8];
u3(2.26486255716695,-3.24754358255360,0.118732689927577) q0[10];
cx q0[10],q0[8];
u1(1.33132340540712) q0[8];
u3(-0.772874776112845,0.0,0.0) q0[10];
cx q0[8],q0[10];
u3(-0.257861045793055,0.0,0.0) q0[10];
cx q0[10],q0[8];
u3(1.33178536373176,2.78211720384759,-1.00719052846276) q0[8];
u3(2.60778430953365,-2.44393134583258,1.02485222212393) q0[10];
barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12];
measure q0[0] -> c0[0];
measure q0[1] -> c0[1];
measure q0[2] -> c0[2];
measure q0[3] -> c0[3];
measure q0[4] -> c0[4];
measure q0[5] -> c0[5];
measure q0[6] -> c0[6];
measure q0[7] -> c0[7];
measure q0[8] -> c0[8];
measure q0[9] -> c0[9];
measure q0[10] -> c0[10];
measure q0[11] -> c0[11];
measure q0[12] -> c0[12];