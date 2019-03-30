echo 'Gen Frames'
python run_parallel.py gen_sens.sh

echo 'Gen OBJs'
#input _vh_clean_2.labels.ply
#output .labels.obj, .labels.txt
python run_parallel.py gen_objs.sh


echo 'Gen Textiles'
#input .labels.obj
#output _textiles002.txt, _frame002.txt
python run_parallel.py gen_textiles.sh


echo 'Gen Labels'
#input _textiles002.txt, .labels.obj, .labels.txt 
#output _labels002.txt, _bary002.txt
python run_parallel.py gen_labels.sh

echo 'Gen Colors'
#input _textiles002.txt, _frame002.txt
#output _color002.txt
python run_parallel.py gen_colors.sh

echo 'Gen Chunks'
python run_parallel.py gen_chunks.sh 