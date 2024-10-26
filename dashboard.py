import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

def load_data():
    """Загрузка и подготовка данных"""
    @st.cache_data
    def _load_data():
        df = pd.read_csv('ar_metrics_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    return _load_data()

def setup_page():
    """Настройка страницы"""
    st.set_page_config(page_title="Interactive AR Analytics Dashboard", layout="wide")
    st.title('Интерактивный дашборд аналитики AR')

def create_filters(df):
    """Создание фильтров"""
    with st.expander("📊 Настройки анализа и фильтры", expanded=True):
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            date_range = st.date_input(
                "Выберите период",
                value=(df['date'].min(), df['date'].max()),
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )

            selected_categories = st.multiselect(
                'Категории',
                options=df['category'].unique(),
                default=df['category'].unique()
            )

        with filter_col2:
            min_users = int(df['active_users'].min())
            max_users = int(df['active_users'].max())
            users_range = st.slider(
                'Диапазон активных пользователей',
                min_users, max_users,
                (min_users, max_users)
            )

            conversion_range = st.slider(
                'Диапазон конверсии (%)',
                0.0, 100.0,
                (0.0, 100.0)
            )

        with filter_col3:
            metrics_to_show = st.multiselect(
                'Выберите метрики для анализа',
                options=['active_users', 'conversion_rate', 'engagement_rate',
                         'retention_rate', 'avg_revenue_per_user', 'total_revenue'],
                default=['active_users', 'conversion_rate', 'total_revenue']
            )

            aggregation = st.selectbox(
                'Тип агрегации данных',
                ['Сумма', 'Среднее', 'Медиана', 'Максимум', 'Минимум']
            )

    return date_range, selected_categories, users_range, conversion_range, metrics_to_show, aggregation

def filter_data(df, date_range, selected_categories, users_range, conversion_range):
    """Применение фильтров к данным"""
    mask = (
            (df['date'].dt.date >= date_range[0]) &
            (df['date'].dt.date <= date_range[1]) &
            (df['category'].isin(selected_categories)) &
            (df['active_users'].between(users_range[0], users_range[1])) &
            (df['conversion_rate'].between(conversion_range[0]/100, conversion_range[1]/100))
    )
    return df[mask]

def aggregate_data(data, agg_type):
    """Агрегация данных"""
    if agg_type == 'Сумма':
        return data.sum()
    elif agg_type == 'Среднее':
        return data.mean()
    elif agg_type == 'Медиана':
        return data.median()
    elif agg_type == 'Максимум':
        return data.max()
    else:
        return data.min()

def display_main_metrics(filtered_df, metrics_to_show, aggregation, date_range):
    """Отображение основных метрик"""
    st.markdown("### Ключевые показатели эффективности (KPI)")
    kpi_cols = st.columns(4)

    for i, metric in enumerate(metrics_to_show[:4]):
        with kpi_cols[i % 4]:
            current_value = aggregate_data(filtered_df[metric], aggregation)
            previous_mask = (
                    (filtered_df['date'].dt.date >= (date_range[0] - timedelta(days=len(filtered_df)))) &
                    (filtered_df['date'].dt.date < date_range[0])
            )
            previous_value = aggregate_data(filtered_df[previous_mask][metric], aggregation)
            delta = ((current_value - previous_value) / previous_value * 100
                     if previous_value != 0 else 0)

            st.metric(
                label=metric.replace('_', ' ').title(),
                value=f"{current_value:.2f}",
                delta=f"{delta:.1f}%"
            )

def create_trend_chart(filtered_df, metrics_to_show):
    """Создание графика трендов с корректной агрегацией по категориям"""
    st.markdown("### Тренды метрик")

    chart_metrics = st.multiselect(
        'Выберите метрики для отображения на графике',
        options=metrics_to_show,
        default=metrics_to_show[0]
    )

    fig = go.Figure()

    # Группируем данные по дате и категории
    for metric in chart_metrics:
        # Для каждой категории создаем отдельную линию
        for category in filtered_df['category'].unique():
            category_data = filtered_df[filtered_df['category'] == category]

            # Агрегируем данные в зависимости от типа метрики
            if metric in ['avg_revenue_per_user', 'conversion_rate', 'engagement_rate',
                          'retention_rate', 'crash_rate', 'load_time_sec']:
                # Для средних значений используем mean()
                agg_data = category_data.groupby('date')[metric].mean()
            else:
                # Для абсолютных значений используем sum()
                agg_data = category_data.groupby('date')[metric].sum()

            fig.add_trace(go.Scatter(
                x=agg_data.index,
                y=agg_data.values,
                name=f"{category} - {metric}".replace('_', ' ').title(),
                mode='lines+markers'
            ))

    fig.update_layout(
        title='Динамика изменения метрик во времени',
        xaxis_title='Дата',
        yaxis_title='Значение',
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def display_financial_analysis(filtered_df):
    """Отображение финансового анализа с корректной агрегацией"""
    st.markdown("### Финансовый анализ")

    revenue_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Агрегируем данные по дате для всех категорий
    daily_revenue = filtered_df.groupby(['date', 'category']).agg({
        'total_revenue': 'sum',
        'avg_revenue_per_user': 'mean'
    }).reset_index()

    # Добавляем общий доход (сумма по всем категориям)
    total_daily_revenue = daily_revenue.groupby('date')['total_revenue'].sum()
    revenue_fig.add_trace(
        go.Bar(
            x=total_daily_revenue.index,
            y=total_daily_revenue.values,
            name="Общий доход"
        ),
        secondary_y=False,
    )

    # Добавляем средний доход на пользователя (среднее по всем категориям)
    avg_revenue_per_user = daily_revenue.groupby('date')['avg_revenue_per_user'].mean()
    revenue_fig.add_trace(
        go.Scatter(
            x=avg_revenue_per_user.index,
            y=avg_revenue_per_user.values,
            name="Средний доход на пользователя",
            line=dict(color='red')
        ),
        secondary_y=True,
    )

    revenue_fig.update_layout(
        title='Анализ доходности',
        xaxis_title="Дата",
        barmode='group',
        hovermode='x unified'
    )

    revenue_fig.update_yaxes(title_text="Общий доход", secondary_y=False)
    revenue_fig.update_yaxes(title_text="Средний доход на пользователя", secondary_y=True)

    st.plotly_chart(revenue_fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Группируем доход по категориям
        category_revenue = filtered_df.groupby('category')['total_revenue'].sum()
        fig_category = px.pie(
            values=category_revenue.values,
            names=category_revenue.index,
            title='Распределение дохода по категориям'
        )
        st.plotly_chart(fig_category, use_container_width=True)

    with col2:
        # Считаем средний доход на пользователя по категориям
        avg_revenue = filtered_df.groupby('category')['avg_revenue_per_user'].mean()
        fig_avg = px.bar(
            x=avg_revenue.index,
            y=avg_revenue.values,
            title='Средний доход на пользователя по категориям'
        )
        fig_avg.update_layout(
            xaxis_title="Категория",
            yaxis_title="Средний доход на пользователя"
        )
        st.plotly_chart(fig_avg, use_container_width=True)

def display_performance_analysis(filtered_df, metrics_to_show):
    """Отображение анализа производительности"""
    st.markdown("### Анализ производительности")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        load_time_fig = px.box(
            filtered_df,
            x='category',
            y='load_time_sec',
            title='Распределение времени загрузки по категориям'
        )
        st.plotly_chart(load_time_fig, use_container_width=True)

    with perf_col2:
        crash_fig = px.line(
            filtered_df.groupby(['date', 'category'])['crash_rate'].mean().reset_index(),
            x='date',
            y='crash_rate',
            color='category',
            title='Динамика частоты сбоев'
        )
        st.plotly_chart(crash_fig, use_container_width=True)

    # Тепловая карта корреляций
    st.markdown("### Корреляционный анализ")
    correlation = filtered_df[metrics_to_show].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation,
        x=correlation.columns,
        y=correlation.columns,
        text=correlation.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu',
        zmid=0
    ))

    fig_corr.update_layout(
        title='Тепловая карта корреляций между метриками',
        xaxis_title="Метрики",
        yaxis_title="Метрики",
        width=800,
        height=800
    )

    st.plotly_chart(fig_corr, use_container_width=True)

def display_raw_data(filtered_df):
    """Отображение сырых данных"""
    with st.expander("📋 Просмотр исходных данных"):
        st.markdown("### Детальные данные")

        columns_to_show = st.multiselect(
            'Выберите колонки для отображения',
            options=filtered_df.columns,
            default=list(filtered_df.columns)
        )

        st.dataframe(
            filtered_df[columns_to_show].style.highlight_max(
                axis=0,
                subset=[col for col in ['active_users', 'total_revenue'] if col in columns_to_show]
            ),
            height=300
        )

        st.download_button(
            label="📥 Скачать отфильтрованные данные",
            data=filtered_df[columns_to_show].to_csv(index=False).encode('utf-8'),
            file_name='filtered_ar_data.csv',
            mime='text/csv'
        )

def main():
    """Основная функция"""
    # Настройка страницы
    setup_page()

    # Загрузка данных
    df = load_data()

    # Создание фильтров
    date_range, selected_categories, users_range, conversion_range, metrics_to_show, aggregation = create_filters(df)

    # Применение фильтров
    filtered_df = filter_data(df, date_range, selected_categories, users_range, conversion_range)

    # Создание вкладок
    tab1, tab2, tab3 = st.tabs(["📈 Основные метрики", "💰 Финансовый анализ", "🎯 Производительность"])

    # Заполнение вкладок
    with tab1:
        display_main_metrics(filtered_df, metrics_to_show, aggregation, date_range)
        create_trend_chart(filtered_df, metrics_to_show)

    with tab2:
        display_financial_analysis(filtered_df)

    with tab3:
        display_performance_analysis(filtered_df, metrics_to_show)

    # Отображение сырых данных
    display_raw_data(filtered_df)

if __name__ == "__main__":
    main()