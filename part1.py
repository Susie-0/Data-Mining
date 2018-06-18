import pandas as pd
import numpy as np
import math

uid_train = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_train/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_train/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

voice_test = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_Test-B/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_Test-B/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('/Users/macbook/Desktop/JDATA用户风险识别_Test-B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('/Users/macbook/Desktop/JDATA用户风险识别_Test-B/uid_test_b.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

voice['start_time'] = pd.to_numeric(voice.start_time, errors='coerce').fillna(0).astype(np.int64)

voice['end_time'] = pd.to_numeric(voice.end_time, errors='coerce').fillna(0).astype(np.int64)

sms['start_time'] = pd.to_numeric(sms.start_time, errors='coerce').fillna(0).astype(np.int64)

def todate( tmp1 ):
    
    tmpa = tmp1%100
    tmpa += math.floor(tmp1%10000/100) * 60
    tmpa += math.floor(tmp1%1000000/10000) * 3600
    tmpa += math.floor(tmp1/1000000) * 86400
    tmpa = int(tmpa)
    return tmpa;

def whathour( time1 ):
    tmp = time1%86400
    tmp = math.floor(tmp/3600)
    tmp = int(tmp)
    return tmp

voice['start_time'] = voice['start_time'].apply(lambda x: todate(x)).fillna(0)

voice['end_time'] = voice['end_time'].apply(lambda x: todate(x)).fillna(0)

voice['long'] = voice['end_time'] - voice['start_time']

voice['hour'] = voice['start_time'].apply(lambda x: whathour(x)).fillna(0)



sms['start_time'] = sms['start_time'].apply(lambda x: todate(x)).fillna(0)

sms['hour'] = sms['start_time'].apply(lambda x: whathour(x)).fillna(0)



vocie_call_hour = voice.groupby(['uid','hour'])['uid'].count().unstack().add_prefix('voice_call_hour_').reset_index().fillna(0)

voice_long = voice.groupby(['uid'])['long'].agg({'mean': 'mean','max':'max'}).add_prefix('voice_long_').reset_index().fillna(0)

voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)


sms_send_hour = sms.groupby(['uid','hour'])['uid'].count().unstack().add_prefix('sms_send_hour_').reset_index().fillna(0)

sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()

sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()


up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()


feature = [vocie_call_hour,voice_long,voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,sms_send_hour,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,wa_name,visit_cnt,visit_dura,up_flow,
           down_flow]

train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid').fillna(0)

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid').fillna(0)

train_feature.to_csv('/Users/macbook/Desktop/train_featureV2.csv',index=None)
test_feature.to_csv('/Users/macbook/Desktop/test_featureV2.csv',index=None)