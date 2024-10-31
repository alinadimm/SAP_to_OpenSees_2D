#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------------Steel Fiber Section Function------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
import openseespy.opensees as ops
import pandas as pd
import opsvis as opsv

#   quad definition style: 
#              ( i        ,  j      ,
#               
#                k        ,  l      )


#--------------------I/W Section---------------------------
def I_sec(secTag,matTag,E,bf,tf,h,tw):
    G=E/2.6
    hw = h-2*tf
    J=1/3*(2*bf*tf**3+h*tw**3) #########
    ops.section('Fiber', secTag, '-GJ', G*J)
   
    ops.patch('quad', matTag, 10, 2,  
              h/2,-bf/2  ,   h/2,bf/2   ,
              hw/2,bf/2,   hw/2,-bf/2)
    
    ops.patch('quad', matTag, 10, 2,  
              -hw/2,-bf/2  , -hw/2,bf/2   ,
              -h/2,bf/2,   -h/2,-bf/2)
    
    ops.patch('quad', matTag, 2, 10,  
              hw/2,-tw/2  , hw/2,tw/2   ,
              -hw/2,tw/2,   -hw/2,-tw/2)


def I_secplot(secTag,matTag,E,bf,tf,h,tw):
    G=E/2.6
    hw = h-2*tf
    J=1/3*(2*bf*tf**3+h*tw**3) #########
    
    
    return [['section','Fiber', secTag, '-GJ', G*J],
   
    ['patch', 'quad', matTag, 10, 2,  
              h/2,-bf/2  ,   h/2,bf/2   ,
              hw/2,bf/2,   hw/2,-bf/2],
    
    ['patch','quad', matTag, 10, 2,  
              -hw/2,-bf/2  , -hw/2,bf/2   ,
              -h/2,bf/2,   -h/2,-bf/2],
    
    ['patch','quad', matTag, 2, 10,  
              hw/2,-tw/2  , hw/2,tw/2   ,
              -hw/2,tw/2,   -hw/2,-tw/2]]

#--------------------BOX Section---------------------------

def Box_sec(secTag,matTag,E,B,H,tf,tw):
    G=E/2.6
    b = B - tw
    h = H - tf
    J= (2*b**2*h**2)/((b/tf)+(h/tw)) #########
    ops.section('Fiber', secTag, '-GJ', G*J)
   
    ops.patch('quad', matTag, 20, 2,  
              H/2,-1*(B/2-tw)  ,   H/2,B/2-tw   ,
              H/2-tf,B/2-tw,   H/2-tf,-1*(B/2-tw))
    
    ops.patch('quad', matTag, 20, 2,  
              -1*(H/2-tf),-1*(B/2-tw)  ,   -1*(H/2-tf),B/2-tw   ,
              -H/2,B/2-tw              ,   -H/2,-1*(B/2-tw))

    ops.patch('quad', matTag, 2, 20,  
              H/2,B/2-tw  ,    H/2,B/2  ,
              -H/2,B/2    ,   -H/2,B/2-tw)
   
    ops.patch('quad', matTag, 2, 20,  
              H/2,-B/2            ,    H/2,-1*(B/2-tw)  ,
              -H/2,-1*(B/2-tw)    ,   -H/2,-B/2)
    
def Box_secplot(secTag,matTag,E,B,H,tf,tw):
    G=E/2.6
    b = B - tw
    h = H - tf
    J= (2*b**2*h**2)/((b/tf)+(h/tw)) #########
    return [['section','Fiber', secTag, '-GJ', G*J],
   
    ['patch' ,'quad', matTag, 20, 2,  
              H/2,-1*(B/2-tw)  ,   H/2,B/2-tw   ,
              H/2-tf,B/2-tw,   H/2-tf,-1*(B/2-tw)],
    
    ['patch' ,'quad', matTag, 20, 2,  
              -1*(H/2-tf),-1*(B/2-tw)  ,   -1*(H/2-tf),B/2-tw   ,
              -H/2,B/2-tw              ,   -H/2,-1*(B/2-tw)],

    ['patch', 'quad', matTag, 2, 20,  
              H/2,B/2-tw  ,    H/2,B/2  ,
              -H/2,B/2    ,   -H/2,B/2-tw],
   
    ['patch' ,'quad', matTag, 2, 20,  
              H/2,-B/2            ,    H/2,-1*(B/2-tw)  ,
              -H/2,-1*(B/2-tw)    ,   -H/2,-B/2]]
   

#--------------------DoubleChannel Section---------------------------


def DoubleChannel_sec(secTag,matTag,E,bf,tf,h,tw,tp):
                        #  tp: Gusset plate thickness
                        #  h : overall height
                        #  bf: single-channel flange width
    G=E/2.6
    hw = h-2*tf
    J=1/3*(2*bf*tf**3+h*tw**3) #########
    ops.section('Fiber', secTag, '-GJ', G*J)
   
    ops.patch('quad', matTag, 10, 2,  
              h/2,tp/2  ,   h/2,tp/2+bf   ,
              hw/2,tp/2+bf,   hw/2,tp/2)
    
    ops.patch('quad', matTag, 2, 10,  
              hw/2,tp/2+bf-tw  , hw/2,tp/2+bf   ,
              -hw/2,tp/2+bf,   -hw/2,tp/2+bf-tw)
    
    ops.patch('quad', matTag, 2, 10,  
              -hw/2,tp/2  , -hw/2,tp/2+bf   ,
              -h/2,tp/2+bf,   -h/2,tp/2)
    
    ops.patch('quad', matTag, 10, 2,  
              h/2,-tp/2  ,   h/2,-(tp/2+bf)   ,
              hw/2,-(tp/2+bf),   hw/2,-tp/2)
    
    ops.patch('quad', matTag, 2, 10,  
              hw/2,-(tp/2+bf-tw)  , hw/2,-(tp/2+bf)   ,
              -hw/2,-(tp/2+bf),   -hw/2,-(tp/2+bf-tw))
   
    
    ops.patch('quad', matTag, 2, 10,  
              -hw/2,-tp/2  , -hw/2,-(tp/2+bf)   ,
              -h/2,-(tp/2+bf),   -h/2,-tp/2)
    
    
def DoubleChannel_secplot(secTag,matTag,E,bf,tf,h,tw,tp):
    G=E/2.6
    hw = h-2*tf
    J=1/3*(2*bf*tf**3+h*tw**3) #########
    return [['section','Fiber', secTag, '-GJ', G*J],
   
    ['patch','quad', matTag, 10, 2,  
              h/2,tp/2  ,   h/2,tp/2+bf   ,
              hw/2,tp/2+bf,   hw/2,tp/2],
    
    ['patch','quad', matTag, 2, 20,  
              hw/2,tp/2+bf-tw  , hw/2,tp/2+bf   ,
              -hw/2,tp/2+bf,   -hw/2,tp/2+bf-tw],
    
    ['patch','quad', matTag, 10, 2,  
              -hw/2,tp/2  , -hw/2,tp/2+bf   ,
              -h/2,tp/2+bf,   -h/2,tp/2],
    
    ['patch','quad', matTag, 10, 2,  
              h/2,-tp/2  ,   h/2,-(tp/2+bf)   ,
              hw/2,-(tp/2+bf),   hw/2,-tp/2],
    
    ['patch','quad', matTag, 2, 20,  
              hw/2,-(tp/2+bf-tw)  , hw/2,-(tp/2+bf)   ,
              -hw/2,-(tp/2+bf),   -hw/2,-(tp/2+bf-tw)],
   
    
    ['patch','quad', matTag, 10, 2,  
              -hw/2,-tp/2  , -hw/2,-(tp/2+bf)   ,
              -h/2,-(tp/2+bf),   -h/2,-tp/2]]


def Cruciform_sec(secTag,matTag,E,bf,tf,h,tw):
    G=E/2.6
    hw = h-2*tf
    J=1/3*(2*bf*tf**3+h*tw**3) #########
    ops.section('Fiber', secTag, '-GJ', G*J)
   
    ops.patch('quad', matTag, 10, 2,  
              h/2,-bf/2  ,   h/2,bf/2   ,
              hw/2,bf/2,   hw/2,-bf/2)
    
    ops.patch('quad', matTag, 2, 20,  
              hw/2,-tw/2  , hw/2,tw/2   ,
              -hw/2,tw/2,   -hw/2,-tw/2)
    
    ops.patch('quad', matTag, 10, 2,  
              -hw/2,-bf/2  , -hw/2,bf/2   ,
              -h/2,bf/2,   -h/2,-bf/2)
    
    ops.patch('quad', matTag, 2, 10,  
              bf/2,hw/2  ,   bf/2,h/2   ,
              -bf/2,h/2,   -bf/2,hw/2)
    
    ops.patch('quad', matTag, 10, 2,  
              tw/2,tw/2  ,   tw/2,hw/2   ,
              -tw/2,hw/2,   -tw/2,tw/2)

    ops.patch('quad', matTag, 10, 2,  
              tw/2,-tw/2  ,   tw/2,-hw/2   ,
              -tw/2,-hw/2,   -tw/2,-tw/2)

    ops.patch('quad', matTag, 2, 10,  
              bf/2,-hw/2  ,   bf/2,-h/2   ,
              -bf/2,-h/2,   -bf/2,-hw/2)
 
       
def Cruciform_secplot(secTag,matTag,E,bf,tf,h,tw):
    G=E/2.6
    hw = h-2*tf
    J=1/3*(2*bf*tf**3+h*tw**3) #########
    return [['section', 'Fiber', secTag, '-GJ', G*J],
   
    ['patch' ,'quad', matTag, 10, 2,  
              h/2,-bf/2  ,   h/2,bf/2   ,
              hw/2,bf/2,   hw/2,-bf/2],
    
    ['patch' ,'quad', matTag, 2, 20,  
              hw/2,-tw/2  , hw/2,tw/2   ,
              -hw/2,tw/2,   -hw/2,-tw/2],
    
    ['patch' ,'quad', matTag, 10, 2,  
              -hw/2,-bf/2  , -hw/2,bf/2   ,
              -h/2,bf/2,   -h/2,-bf/2],
    
    ['patch' ,'quad', matTag, 2, 10,  
              bf/2,hw/2  ,   bf/2,h/2   ,
              -bf/2,h/2,   -bf/2,hw/2],
    
    ['patch' ,'quad', matTag, 10, 2,  
              tw/2,tw/2  ,   tw/2,hw/2   ,
              -tw/2,hw/2,   -tw/2,tw/2],

    ['patch' ,'quad', matTag, 10, 2,  
              tw/2,-tw/2  ,   tw/2,-hw/2   ,
              -tw/2,-hw/2,   -tw/2,-tw/2],

    ['patch','quad', matTag, 2, 10,  
              bf/2,-hw/2  ,   bf/2,-h/2   ,
              -bf/2,-h/2,   -bf/2,-hw/2]]
          