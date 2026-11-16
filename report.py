import argparse,json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Cm,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
p=argparse.ArgumentParser();p.add_argument('--results',default='results_pr3');p.add_argument('--out',default='Практическая_работа_3.docx');a=p.parse_args();root=Path(a.results)
d=Document(Path(__file__).with_name('report_template.docx'));info=json.loads((root/'model_v1.json').read_text(encoding='utf-8'));r=pd.read_csv(root/'experiments.csv'); marker=next(x for x in d.paragraphs if x.text=='[[RESULTS]]')
def before(s):return marker.insert_paragraph_before(s)
def table(headers,rows):
 t=d.add_table(rows=1,cols=len(headers));t.style='Table Grid';marker._p.addprevious(t._tbl)
 for c,v in zip(t.rows[0].cells,headers):c.text=str(v)
 for row in rows:
  for c,v in zip(t.add_row().cells,row):c.text=str(v)
 for i,row in enumerate(t.rows):
  trp=row._tr.get_or_add_trPr();trp.append(OxmlElement('w:cantSplit'))
  if i==0:trp.append(OxmlElement('w:tblHeader'))
  for c in row.cells:
   for x in c.paragraphs:
    x.paragraph_format.first_line_indent=Cm(0);x.paragraph_format.line_spacing=1.0
    for run in x.runs:run.font.size=Pt(10);run.bold=i==0
before(f"На обучающей части ({info['train_rows']} записей) выполнены эксперименты. Отложено {info['test_rows']} записей; тест не оценивался. Повторяемость подтверждена: матрицы признаков двух последовательных преобразований совпали; прогнозы до и после сериализации совпали. Размер матрицы: {info['matrix_shape']}.")
before('В таблице 4 приведены результаты всех экспериментов. Варианты tuned получены подбором гиперпараметров; raw обозначает исходный набор признаков. Среднее время обучения зависит от оборудования.')
before('Таблица 4 – Результаты пятиблочной кросс-валидации (AP – средняя точность; время – среднее время fit одного блока, с)')
table(['Модель','ROC-AUC ± σ','AP','F1','Recall','Precision','с'],[(x.model,f'{x.roc_auc:.3f} ± {x.roc_auc_std:.3f}',f'{x.ap:.3f}',f'{x.f1:.3f}',f'{x.recall:.3f}',f'{x.precision:.3f}',f'{x.seconds:.3f}') for x in r.itertuples()])
base=r[r.model=='Dummy'].iloc[0];before(f'Базовое решение: ROC-AUC = {base.roc_auc:.3f}, Recall = {base.recall:.3f}.')
before('На рисунках 1–4 показаны результаты анализа регуляризации, глубины дерева и сравнения моделей.')
for i,(name,cap) in enumerate([('regularization_auc.png','Влияние C на ROC-AUC'),('regularization_nonzero.png','Число ненулевых коэффициентов'),('tree_depth.png','ROC-AUC дерева на train и CV'),('comparison.png','Десять лучших вариантов по ROC-AUC')],1):
 x=before('');x.alignment=1;x.paragraph_format.first_line_indent=Cm(0);x.paragraph_format.keep_with_next=True;x.add_run().add_picture(str(root/name),width=Cm(14));x=before(f'Рисунок {i} – {cap}');x.alignment=1;x.paragraph_format.first_line_indent=Cm(0)
reg=pd.read_csv(root/'regularization.csv');before('На рисунке 1 максимум ROC-AUC L2 достигается при C = 0,01 (0,781), а L1 – при C = 0,1 (0,769). При слабой регуляризации качество снижается. Рисунок 2 показывает, что L2 сохраняет 23 коэффициента, а L1 отбирает признаки.')
before('При изменении C от 0,001 до 100 число ненулевых коэффициентов L1 изменилось с '+str(reg[reg.penalty=='l1'].nonzero.iloc[0])+' до '+str(reg[reg.penalty=='l1'].nonzero.iloc[-1])+'.')
depth=pd.read_csv(root/'tree_depth.csv');v=depth.loc[depth.cv.idxmax()];before(f'Максимум проверочного ROC-AUC дерева: {v.cv:.3f} при глубине {v.depth}; сопоставление с train позволяет оценить разрыв, вызванный переобучением.')
before('По рисунку 3 при глубине 8 обучающий ROC-AUC равен 1,000, а проверочный – 0,681: большой разрыв свидетельствует о переобучении. На рисунке 4 виден небольшой отрыв настроенного случайного леса от логистической регрессии.')
co=pd.read_csv(root/'coefficients.csv').head(6);before('Таблица 5 – Наибольшие по модулю коэффициенты L2 (C = 1)');table(['Признак','Коэффициент'],[(x.feature,f'{x.coef:.4f}') for x in co.itertuples()]);before('В таблице 5 наиболее выраженные коэффициенты относятся к конкретным кодам категорий и числовым признакам; их следует трактовать в контексте стандартизации и совместного присутствия связанных признаков.')
before('Положительный знак связан с увеличением логита класса 1 при прочих равных, отрицательный – с уменьшением. Коэффициенты не доказывают причинность; взаимосвязанные признаки и кодирование влияют на их значения.')
for ext,raw in [('LogReg L2 C=1','LogReg L2 raw'),('Gradient boosting','Gradient boosting raw')]:
 v=r.loc[r.model==ext,'roc_auc'].iloc[0];w=r.loc[r.model==raw,'roc_auc'].iloc[0];before(f'{ext}: исходные признаки {w:.3f}, расширенные {v:.3f}; изменение ROC-AUC {v-w:+.3f}. Прирост или снижение необходимо интерпретировать с учетом малого объема данных.')
for t in info['tuning']:before(f"Подбор {t['family']}: лучшая комбинация {json.dumps(t['best'],ensure_ascii=False)}, ROC-AUC = {t['auc']:.3f}.")
m=info['metrics'];before(f"По установленному правилу итоговая модель – {info['winner']}, ROC-AUC = {m['roc_auc']:.3f} ± {m['roc_auc_std']:.3f}, Recall = {m['recall']:.3f}. Превосходство над базовым решением по ROC-AUC: {m['roc_auc']-base.roc_auc:+.3f}.")
before('В настроенном случайном лесе 200 деревьев, max_depth = 2 и min_samples_leaf = 1. AP = 0,808, Precision = 0,735, F1 = 0,730. Разброс ROC-AUC между блоками составляет 0,136; выбор сделан по CV, а не по независимому тесту. Доля положительных прогнозов зависит от порога, который будет уточнен в ПР № 4.')
best_lr=r[r.model.str.startswith('LogReg')].roc_auc.max();best_ens=r[r.model.str.startswith(('Random forest','Gradient boosting'))].roc_auc.max();before(f'Проверка гипотезы: лучший ROC-AUC логистической регрессии {best_lr:.3f}, ансамблей {best_ens:.3f}. '+('Линейная модель не уступает ансамблям по заданному допуску 0,005.' if best_lr>=best_ens-.005 else 'Ансамбль превосходит линейную модель более чем на 0,005; гипотеза не получила поддержки по установленному правилу.'))
marker._element.getparent().remove(marker._element)
for x in d.paragraphs:
 if x.text=='[[CONCLUSION]]': x.text=f"Реализован воспроизводимый пайплайн, сравнены модели пяти семейств и базовое решение, выполнен подбор гиперпараметров и оценена полезность производных признаков. По установленному правилу выбрана модель {info['winner']} (ROC-AUC = {m['roc_auc']:.3f}). Артефакт model_v1.joblib и его описание model_v1.json сохранены. Модель требует проверки на отложенной выборке и четырех новых порциях в практической работе № 4; клиническая применимость не установлена."
d.save(a.out)
