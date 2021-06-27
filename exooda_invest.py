import dash_core_components as dcc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_table

#for tables
import numpy as np
import pandas as pd
import camelot.io as cmt
import re
import numpy_financial as np_fin
from sklearn.preprocessing import MinMaxScaler as MMS
import tabula as tbl

invest_pat = r"C:\Users\bseot\Documents\EXOODA\INVESTEC\91-IP-TFSA-List-of-Funds-en.pdf"

df_invest= cmt.read_pdf(
    invest_pat,
    pages='1-8',
    password=None,
    flavor='stream',
    suppress_stdout=False,
#     layout_kwargs={},
#     **kwargs,
)


# In[4]:


invest_column_names = ['FUND NAME', 'CLASS', 'TOTAL ANNUAL FEE', 'FOREIGN EXPOSURE', 'EQUITY EXPOSURE', 'PROPERTY EXPOSURE', 'Annual Growth Rate']
df_large_invest = df_invest[2].df.drop([0,1,2])
for table in df_invest[3:]:
    tab = table.df
    tab = tab.drop([0,1,2], axis = 0)
    df_large_invest = pd.concat([df_large_invest,tab],axis = 0)
np.random.seed(97)
df_large_invest['Annual Growth Rate'] = np.random.normal(loc = 0.06, scale = 0.02, size = len(df_large_invest))
df_large_invest.columns = invest_column_names


# In[5]:


df_large_invest = df_large_invest.reset_index()
df_large_invest = df_large_invest.drop('index', axis = 1)
df_large_invest = df_large_invest.drop('CLASS', axis = 1)
df_large_invest = df_large_invest[df_large_invest['TOTAL ANNUAL FEE'] != '']


# In[6]:


df_large_invest_crude = df_large_invest.copy()
df_large_invest_crude = df_large_invest_crude.reset_index(drop = True)
cols = df_large_invest.drop('FUND NAME', axis = 1).columns
df_large_invest[cols[:-1]] = df_large_invest[cols[:-1]].applymap(lambda x: x.replace('%',''))
df_large_invest[cols] = df_large_invest[cols].apply(pd.to_numeric, errors='coerce')
df_large_invest = df_large_invest.reset_index(drop = True)


# In[7]:


df_large_invest_crude


# In[8]:


df_large_invest['TOTAL ANNUAL FEE'] = (df_large_invest['TOTAL ANNUAL FEE']-100)*-1


# # Ranking

# In[9]:


df_large_invest['Annual Growth Rate (%)'] = df_large_invest['Annual Growth Rate']
# df_large_invest = df_large_invest.drop('Annual Growth Rate', axis = 1)
df_large_invest[cols] = MMS().fit_transform(df_large_invest[cols])
df_large_invest[cols] = df_large_invest[cols]*100
df_large_invest[cols] = round(df_large_invest[cols], 2)


# In[10]:


df_large_invest[cols] = round(df_large_invest[cols].rank(numeric_only=True, axis = 1, method = 'min')).astype(int)


# In[11]:


df_large_invest['e Score'] = df_large_invest[cols].sum(axis = 1)

fee = 'On a scale of 1-10, how important are fees to you (1 = Not concerned)?'
fex = 'How much foreign exposure would you like the investment to have  in percentage terms)?'
eqx = 'How much equity exposure would you like the investment to have (in percentage terms)?'
prx = 'How much property exposure would you like the investment to have (in percentage terms)?'
princ = 'What is your initial deposit amount?'
insta = 'What will your monthly installment be?'
n = 'For how many years would you like to keep this investment?'
mat = 'To what amount would you like your investment to grow?'

style_dicc = {'backcolor1': '#FFFFFF', 
              'backcolor2': '#F2F2F2',
              'backcolor3': '#D9D9D9',
              'emphasiscolor1': '#991500', 
              'font' : 'Arial'}

mark_text = '''
            #### Waterfall analysis - 9 Months Ending June 2020:
            Please select a tab on the left of the screen to go to each analysis.
            '''
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'borderTop': '1px solid #d6d6d6',
    'borderRight': '1px solid #d6d6d6',
    'borderLeft': '1px solid #d6d6d6',
    'padding': '4px 4px 0px 0px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'padding': '4px',
    'fontWeight': 'bold'
}

external_stylesheets = [dbc.themes.BOOTSTRAP] ## Bootstrrap helps us using columns and rows so everything looks better
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [

dbc.Row([
                                 dbc.Col([
                                        html.H6(fee, style = {'font-size':15}),
                                        dcc.Input(
                                            id="fee", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(fex, style = {'font-size':15}),
                                        dcc.Input(
                                            id="fex", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(eqx, style = {'font-size':15}),
                                        dcc.Input(
                                            id="eqx", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(prx, style = {'font-size':15}),
                                        dcc.Input(
                                            id="prx", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(princ, style = {'font-size':15}),
                                        dcc.Input(
                                            id="princ", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(insta, style = {'font-size':15}),
                                        dcc.Input(
                                            id="insta", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(n, style = {'font-size':15}),
                                        dcc.Input(
                                            id="n", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            ),
                                        html.H6(mat, style = {'font-size':15}),
                                        dcc.Input(
                                            id="mat", type="number", placeholder="Please answer the above question",
                                            min=0, max=1000000000, step=1, value = 0
                                            )                                                                 
                                         ], width = {'size': 2}),
                                 dbc.Col([
                                         html.Div(id = 'questions_matrix'), 
                                         ], width = {'size': 10}),                                                         
                                 ]) 
    
    ]
)


# @app.callback(
#     Output("number-out", "children"),
#     [Input("input_range", "value")]
# )
# def number_render(rangeval):
#     return "range: {}".format(rangeval)
    
                               
@app.callback(
    Output(component_id = 'questions_matrix', component_property = 'children'),
    [Input(component_id = 'fee', component_property = 'value'),
    Input(component_id = 'fex', component_property = 'value'),
    Input(component_id = 'eqx', component_property = 'value'),
    Input(component_id = 'prx', component_property = 'value'),
    Input(component_id = 'princ', component_property = 'value'),
    Input(component_id = 'insta', component_property = 'value'),
    Input(component_id = 'n', component_property = 'value'),
    Input(component_id = 'mat', component_property = 'value')]
)

def questions_matrix(fee, fex, eqx, prx, princ, insta, n, mat):    
    
    fee = int(round(int(fee)/2, 0))

    fex = int(round(int(fex)/2, 0)/10)

    eqx = int(round(int(eqx)/2, 0)/10)

    prx = int(round(int(prx)/2, 0)/10)

    princ = int(princ)

    insta = int(insta)

    n = int(n)

    mat = int(mat)
    



    irate = np_fin.rate(nper = n*12, pmt = insta, pv = princ, fv = -mat, when='end', guess=None, tol=None, maxiter=100)


    irate = irate*12


    # In[15]:


    ins = [fee, fex, eqx, prx, irate]
    diff_cols = ['TOTAL ANNUAL FEE', 'FOREIGN EXPOSURE', 'EQUITY EXPOSURE', 'PROPERTY EXPOSURE', 'Annual Growth Rate (%)']

    user_diffs = df_large_invest.copy()
    for i, col_name in enumerate(diff_cols):
        user_diffs[col_name] = np.abs((user_diffs[col_name] - ins[i]))
        
    user_diffs['e Score'] = user_diffs[diff_cols].sum(axis = 1)
    user_diffs['e Score'] = np.abs(21 - user_diffs['e Score'])


    # In[18]:


    user_final = user_diffs.sort_values(by = 'e Score', ascending = False).iloc[:6,:]


    # In[20]:


    df_large_invest_crude['e Score'] = user_diffs['e Score']
    user_chosen_final_df = df_large_invest_crude.iloc[user_final.index.values]
    
    
    # return user_chosen_final_df.to_html()
    
    return dash_table.DataTable(
                    id='table_2',
                    columns=[{"name": i, "id": i} for i in user_chosen_final_df.columns],                    
                    data = user_chosen_final_df.to_dict('records'),
                    style_table = {'overflowX': 'scroll'},
                    style_header={'backgroundColor': style_dicc['backcolor3'],
                                  'fontWeight': 'bold'}
                    # style_data_conditional = [{'if': {'filter_query': '{Price per Item (%)} eq "100 %"'},
                    #                           'backgroundColor': '#FBF2BA'}]
                    )           
                    
                        
if __name__ == "__main__":
    app.run_server(host="0.0.0.0",port="8050", debug=False,dev_tools_ui=False,dev_tools_props_check=False)