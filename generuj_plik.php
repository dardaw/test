<?php

$plik = file('http://www.mbnet.com.pl/dl.txt');
//6959. 09.11.2023 4,7,24,35,36,47
$lista = [];
$lista[0] = ['Game', 'Date', 'A', 'B', 'C', 'D', 'E', 'F'];
foreach ($plik as $klucz => $linia) {
    list($nr, $data, $liczby) = explode(' ', $linia);
    $liczba = explode(',', $liczby);
    $data = str_replace('.', '/', $data);
    $lista[$klucz + 1] = [str_replace('.', '', $nr), $data, $liczba[0], $liczba[1], $liczba[2], $liczba[3], $liczba[4], (int)$liczba[5]];
}


$fp = fopen('file.csv', 'w');

foreach ($lista as $key => $fields) {
   
    fputcsv($fp, $fields);
}

fclose($fp);
?>