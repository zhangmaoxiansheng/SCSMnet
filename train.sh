#python main.py --log_dir /home/jianing/hdd/runs/DSR_test1
python main.py --epochs 10 --stage distill --loadmodel ./weight/checkpoint_10_2954.tar --savemodel ./distill_weight --learning_rate 1e-4 --log_dir /home/jianing/hdd/runs/DSR_test2
python main.py --epochs 10 --stage distill --loadmodel ./distill_weight/checkpoint_10_2954.tar --savemodel ./distill_weight2 --learning_rate 1e-5 --log_dir /home/jianing/hdd/runs/DSR_test3 --mask_disp 0.5
python main.py --epochs 10 --stage distill --loadmodel ./distill_weight2/checkpoint_10_2954.tar --savemodel ./distill_weight3 --learning_rate 1e-5 --log_dir /home/jianing/hdd/runs/DSR_test4 --mask_disp 0.15