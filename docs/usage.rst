Használat
=========

Szükséges modulok
-----------------
A következő modulok szükségesek a script használatához

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Csomag
     - Verzió
   * - `python <https://www.python.org/>`_
     - 3.9
   * - `pygame <https://www.pygame.org/>`_
     - 2.0
   * - `stable baselines3 <https://stable-baselines3.readthedocs.io/>`_
     - 1.3

Ezek benne vanak a **requirements.txt**-ben is
Ezeket telepíteni lehet a **py -m pip install -r requirements.txt** parancsal

Modelek
-------
A :ref:`teach <teach>` moudlban lehet a model tanitását végezni.
:py:meth:`teach.main` fügvény használatával lehet be tanítani.
A modeleket a **./model** mappában lehet találni

Játék inditása
--------------

A játékod lehet a :py:mod:`Snake` modul futatásával vagy egy pédányád a :py:meth:`Snake.Snake2.play`
A kör inditása előtt kell hetet választani a játékost, pálya méretét és a nehézséget

Irányítás
---------

A kígyó irányitása a nyilakkal történik. Ha ha a játékost típus AI akkor + és a - gombokkal lehet szabálzozni a sebeséget
