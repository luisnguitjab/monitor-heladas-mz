import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Importación del backend modularizado ampliado con Localidades
# from backend import FINCAS_MENDOZA, obtener_datos_clima, ejecutar_backend_analitico
from backend import FINCAS_MENDOZA, ejecutar_backend_analitico, obtener_datos_clima

# ==========================================
# Inicialización de la aplicación Dash
# ==========================================
app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# ==========================================
# Diseño de la interfaz (alto contraste oscuro)
# ==========================================
app.layout = html.Div(style={
    'backgroundColor': '#090d16', 
    'fontFamily': 'sans-serif', 
    'padding': '15px',
    'minHeight': '100vh',
    'color': '#f8fafc'
}, children=[
    
    html.Div(style={'textAlign': 'center', 'marginBottom': '25px', 'padding': '10px'}, children=[
        html.H1("Monitor heladas@mz", 
                style={'color': '#38bdf8', 'fontWeight': '700', 'fontSize': 'calc(18px + 1.5vw)', 'margin': '0 0 10px 0'}),
        html.H1("Sistema para soporte de decisiones (DSS) y analítica predictiva de heladas en Mendoza", 
                style={'color': '#38bdf8', 'fontWeight': '300', 'fontSize': 'calc(18px + 1.5vw)', 'margin': '0 0 10px 0'}),
        html.P("Localidades, backend estadístico y aprendizaje profundo (DL)", 
                style={'color': '#94a3b8', 'fontWeight': 'bold', 'fontSize': 'calc(14px + 0.3vw)'})
    ]),
    
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'marginBottom': '20px'}, children=[
        
        html.Div(style={
            'flex': '1 1 250px', 'backgroundColor': '#1e293b', 'padding': '20px', 
            'borderRadius': '8px', 'border': '2px solid #38bdf8', 'boxSizing': 'border-box'
        }, children=[
            html.Label("Seleccione la Localidad:", style={'fontWeight': '900', 'color': '#f8fafc'}),
            dcc.Dropdown(
                id='dropdown-finca',
                options=[{'label': k, 'value': k} for k in FINCAS_MENDOZA.keys()],
                value='Gualtallary (Tupungato)', 
                clearable=False,
                style={'marginTop': '10px', 'color': '#0f172a', 'fontWeight': 'bold'}
            )
        ]),
        
        html.Div(id='kpi-temperatura', style={'flex': '1 1 200px', 'boxSizing': 'border-box'}),
        html.Div(id='kpi-riesgo', style={'flex': '1 1 200px', 'boxSizing': 'border-box'}),
        html.Div(id='kpi-viento-horas', style={'flex': '1 1 200px', 'boxSizing': 'border-box'}),
        html.Div(id='kpi-suelo-et0', style={'flex': '1 1 200px', 'boxSizing': 'border-box'})
    ]),
    
    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
        html.Div(style={'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'overflow': 'hidden'}, children=[
            dcc.Graph(id='grafico-curva-enfriamiento', config={'displayModeBar': False, 'displaylogo': False})
        ]),
        html.Div(style={'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'overflow': 'hidden'}, children=[
            dcc.Graph(id='grafico-deep-learning', config={'displayModeBar': False, 'displaylogo': False})
        ]),
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
            html.Div(style={'flex': '1 1 400px', 'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'overflow': 'hidden'}, children=[
                dcc.Graph(id='grafico-distribucion-nubes', config={'displayModeBar': False, 'displaylogo': False})
            ]),
            html.Div(style={'flex': '1 1 400px', 'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'overflow': 'hidden'}, children=[
                dcc.Graph(id='grafico-matriz-correlacion', config={'displayModeBar': False, 'displaylogo': False})
            ])
        ])
    ])
])

# ==========================================
# Callbacks y lógica de interacción
# ==========================================
@app.callback(
    [Output('kpi-temperatura', 'children'),
     Output('kpi-riesgo', 'children'),
     Output('kpi-viento-horas', 'children'),
     Output('kpi-suelo-et0', 'children'),
     Output('grafico-curva-enfriamiento', 'figure'),
     Output('grafico-deep-learning', 'figure'),
     Output('grafico-distribucion-nubes', 'figure'),
     Output('grafico-matriz-correlacion', 'figure')],
    [Input('dropdown-finca', 'value')]
)
def actualizar_dashboard(nombre_finca):
    try:
        finca = FINCAS_MENDOZA[nombre_finca]
        umbral = finca['umbral']
        cultivo = finca['cultivo']
        localidad = finca['localidad']
        
        df = obtener_datos_clima(finca['lat'], finca['lon'])
        if df.empty:
            raise ValueError("La API no retornó registros.")

        df = ejecutar_backend_analitico(df, umbral)

        df['riesgo_helada'] = np.select(
            [
                (df['temp_aire'] <= umbral) | (df['prob_helada_dl'] > 75),
                (df['temp_aire'] <= (umbral + 2.0)) & (df['nubosidad'] < 25),
            ],
            ['CRÍTICO', 'ALTO'],
            default='BAJO'
        )
        
        horas_bajo_umbral = len(df[df['temp_aire'] <= umbral])
        
        # Extracción de la fecha y hora del registro actual (primer registro)
        fecha_hora_actual = str(df['fecha_hora'].iloc[0])
        
        temp_actual = df['temp_aire'].iloc[0]
        riesgo_actual = df['riesgo_helada'].iloc[0]
        vel_viento_actual = df['vel_viento'].iloc[0]
        hum_suelo_actual = df['hum_suelo'].iloc[0]
        et0_actual = df['evapotranspiracion'].iloc[0]
        prob_dl_actual = round(df['prob_helada_dl'].iloc[0], 1)
        
        status_viento = "Apto" if vel_viento_actual <= 10 else "Excesivo"
        color_viento_texto = "#4ade80" if vel_viento_actual <= 10 else "#f87171"
        
        if riesgo_actual == 'CRÍTICO':
            color_card_riesgo = '#7f1d1d'
            borde_card_riesgo = '#ef4444'
        elif riesgo_actual == 'ALTO':
            color_card_riesgo = '#7c2d12'
            borde_card_riesgo = '#f97316'
        else:
            color_card_riesgo = '#064e3b'
            borde_card_riesgo = '#22c55e'
        
        # Tarjeta 1: Temperatura (con fecha y hora)
        # =========================================
        card_temp = html.Div(style={'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'textAlign': 'center', 'height': '100%'}, children=[
            html.H3(f"Localidad: {localidad}", style={'margin': '0', 'color': '#38bdf8', 'fontSize': '12px', 'fontWeight': 'bold'}),
            html.P(f"📅 {fecha_hora_actual}", style={'color': '#cbd5e1', 'margin': '2px 0 4px 0', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.H4(cultivo, style={'color': '#f8fafc', 'margin': '4px 0', 'fontSize': '13px', 'fontWeight': '900'}),
            html.H2(f"{temp_actual} °C", style={'color': '#38bdf8', 'fontSize': '24px', 'margin': '4px 0', 'fontWeight': '900'})
        ])
        
        # Tarjeta 2: Riesgo (con fecha y hora)
        # ====================================
        card_riesgo = html.Div(style={'backgroundColor': color_card_riesgo, 'padding': '15px', 'borderRadius': '8px', 'border': f'2px solid {borde_card_riesgo}', 'textAlign': 'center', 'height': '100%'}, children=[
            html.H3("Riesgo (IA / DL)", style={'margin': '0', 'color': '#f8fafc', 'opacity': '0.9', 'fontSize': '13px', 'fontWeight': 'bold'}),
            html.P(f"📅 {fecha_hora_actual}", style={'color': '#f1f5f9', 'margin': '2px 0 4px 0', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.H2(riesgo_actual, style={'color': '#ffffff', 'fontSize': '24px', 'fontWeight': '900', 'margin': '4px 0'}),
            html.P(f"Prob. red neuronal: {prob_dl_actual}%", style={'color': '#cbd5e1', 'margin': '0', 'fontSize': '11px', 'fontWeight': 'bold'})
        ])
        
        # Tarjeta 3: Defensa activa (con fecha y hora)
        # ============================================
        card_viento = html.Div(style={'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'textAlign': 'center', 'height': '100%'}, children=[
            html.H3("Defensa activa", style={'margin': '0', 'color': '#94a3b8', 'fontSize': '13px', 'fontWeight': 'bold'}),
            html.P(f"📅 {fecha_hora_actual}", style={'color': '#cbd5e1', 'margin': '2px 0 4px 0', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.H4(f"Viento: {vel_viento_actual} km/h ({status_viento})", style={'color': color_viento_texto, 'margin': '4px 0', 'fontSize': '13px', 'fontWeight': '900'}),
            html.P(f"Proyección crítica: {horas_bajo_umbral} hs", style={'color': '#f8fafc', 'fontWeight': '900', 'margin': '0', 'fontSize': '12px'})
        ])
        
        # Tarjeta 4: Suelo y demanda (con fecha y hora)
        # =============================================
        card_suelo = html.Div(style={'backgroundColor': '#1e293b', 'padding': '15px', 'borderRadius': '8px', 'border': '2px solid #334155', 'textAlign': 'center', 'height': '100%'}, children=[
            html.H3("Suelo y demanda", style={'margin': '0', 'color': '#94a3b8', 'fontSize': '13px', 'fontWeight': 'bold'}),
            html.P(f"📅 {fecha_hora_actual}", style={'color': '#cbd5e1', 'margin': '2px 0 4px 0', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.H4(f"Humedad suelo: {int(hum_suelo_actual*100)}%", style={'color': '#60a5fa', 'margin': '4px 0', 'fontSize': '14px', 'fontWeight': '900'}),
            html.P(f"Evapotranspiración: {et0_actual} mm/h", style={'color': '#f8fafc', 'margin': '0', 'fontSize': '12px', 'fontWeight': 'bold'})
        ])
        
        grid_style = dict(showgrid=True, gridcolor='#334155', zerolinecolor='#475569', tickfont=dict(color='#f8fafc', size=11))
        layout_oscuro_base = dict(
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font=dict(color='#f8fafc'),
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=45, r=15, t=50, b=40)
        )
        
        # Gráfico 1: Curva térmica con tendencia estadística
        # ==================================================
        fig_curva = go.Figure()
        fig_curva.add_trace(go.Scatter(x=df['fecha_hora'], y=df['temp_aire'], name='Temp. real (aire 2m)', line=dict(color='#38bdf8', width=2.5)))
        fig_curva.add_trace(go.Scatter(x=df['fecha_hora'], y=df['temp_tendencia_estadistica'], name='Pronóstico estadístico (Poly)', line=dict(color='#facc15', width=2.5, dash='dash')))
        fig_curva.add_trace(go.Scatter(x=df['fecha_hora'], y=df['punto_rocio'], name='Punto de rocío', line=dict(color='#c084fc', width=2)))
        fig_curva.add_trace(go.Scatter(x=[df['fecha_hora'].min(), df['fecha_hora'].max()], y=[umbral, umbral], name='Umbral crítico', mode='lines', line=dict(color='#ef4444', width=3, dash='dot')))
        
        fig_curva.update_layout(
            **layout_oscuro_base,
            title=dict(text=f"Dinámica térmica y pronóstico estadístico en: {nombre_finca}", font=dict(color='#f8fafc', size=16, weight='bold')),
            xaxis=dict(title=dict(text="Fecha y hora", font=dict(color='#f8fafc')), **grid_style),
            yaxis=dict(title=dict(text="Temperatura (°C)", font=dict(color='#f8fafc')), **grid_style),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#f8fafc', size=11))
        )
        
        # Gráfico 2: Deep Learning
        # ========================
        fig_dl = go.Figure()
        fig_dl.add_trace(go.Scatter(
            x=df['fecha_hora'], y=df['prob_helada_dl'], 
            name='Prob. Helada Crítica (Deep Learning ANN)', 
            fill='tozeroy', line=dict(color='#f87171', width=3),
            fillcolor='rgba(248, 113, 113, 0.2)'
        ))
        fig_dl.update_layout(
            **layout_oscuro_base,
            title=dict(text="Salida del backend DL: Probabilidad de daño en el tiempo (%)", font=dict(color='#f8fafc', size=16, weight='bold')),
            xaxis=dict(title=dict(text="Fecha y hora", font=dict(color='#f8fafc')), **grid_style),
            yaxis=dict(title=dict(text="Probabilidad (%)", font=dict(color='#f8fafc')), range=[0, 105], **grid_style),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#f8fafc', size=11))
        )
        
        # Gráfico 3: Nubosidad
        # ====================
        fig_nubes = px.bar(df, x='fecha_hora', y='nubosidad', title='Porcentaje de Nubosidad Estimada', color='nubosidad', color_continuous_scale='Teal')
        fig_nubes.update_layout(
            **layout_oscuro_base,
            title=dict(text="Porcentaje de nubosidad estimada", font=dict(color='#f8fafc', size=16, weight='bold')),
            xaxis=dict(title=dict(text="Fecha y hora", font=dict(color='#f8fafc')), **grid_style),
            yaxis=dict(title=dict(text="% Cobertura nubosa", font=dict(color='#f8fafc')), **grid_style),
            coloraxis_showscale=True
        )

        # Gráfico 4: Matriz de correlación
        # ================================
        columnas_interes = ['temp_aire', 'punto_rocio', 'vel_viento', 'temp_suelo', 'evapotranspiracion', 'prob_helada_dl']
        df_corr = df[columnas_interes].corr()
        df_corr.columns = ['Temp. aire', 'Pto. rocío', 'Vel. viento', 'Temp. suelo', 'ET0', 'Prob. DL']
        df_corr.index = df_corr.columns
        
        fig_corr = px.imshow(
            df_corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", 
            title="Matriz de correlación analítica", labels=dict(color="Coeficiente")
        )
        fig_corr.update_layout(
            **layout_oscuro_base,
            title=dict(text="Matriz de correlación analítica", font=dict(color='#f8fafc', size=16, weight='bold')),
            xaxis=dict(tickfont=dict(color='#f8fafc', size=11, weight='bold')),
            yaxis=dict(tickfont=dict(color='#f8fafc', size=11, weight='bold'))
        )

        return card_temp, card_riesgo, card_viento, card_suelo, fig_curva, fig_dl, fig_nubes, fig_corr

    except Exception as e:
        print(f"\n[!!!] ERROR EN EL CALLBACK: {e}\n")
        error_card = html.Div(children=[html.H3("Error", style={'color':'#f87171'}), html.P(str(e), style={'color': '#f8fafc'})])
        fig_vacia = go.Figure().update_layout(paper_bgcolor='#1e293b', plot_bgcolor='#1e293b', font=dict(color='#f8fafc'), title="Datos no disponibles")
        return error_card, error_card, error_card, error_card, fig_vacia, fig_vacia, fig_vacia, fig_vacia

# ==========================================
# Inicialización del servidor
# ==========================================
if __name__ == '__main__':
    app.run(debug=False, port=8050)