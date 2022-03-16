"""
    Example of combining the qtm package with Qt
    Requires PyQt5 and quamash
    Use pip to install requirements:
        pip install -r requirements.txt

    Only tested on Windows, get_interfaces() needs alternative implementation for other platforms
"""

import sys
import asyncio
import subprocess
import re
import xml.etree.cElementTree as ET

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, QObject, pyqtProperty
from PyQt5 import uic

import qtm
from qtm import QRTEvent
from quamash import QSelectorEventLoop

main_window_class, _ = uic.loadUiType("./ui/main.ui")


def start_async_task(task):
    asyncio.ensure_future(task)


def get_interfaces():
    result = subprocess.check_output("ipconfig /all").decode("utf-8")
    result = result.splitlines()
    for line in result:
        if "IPv4" in line:
            found = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", line)
            for ip in found:
                yield ip


class QDiscovery(QObject):
    discoveringChanged = pyqtSignal(bool)
    discoveredQTM = pyqtSignal(str, str)

    def __init__(self, *args):
        super().__init__(*args)
        self._discovering = False
        self._found_qtms = {}

    @pyqtProperty(bool, notify=discoveringChanged)
    def discovering(self):
        return self._discovering

    @discovering.setter
    def discovering(self, value):
        if value != self._discovering:
            self._discovering = value
            self.discoveringChanged.emit(value)

    def discover(self):
        self.discovering = True

        self._found_qtms = {}
        for interface in get_interfaces():
            start_async_task(self._discover_qtm(interface))

    async def _discover_qtm(self, interface):

        try:
            async for qtm_instance in qtm.Discover(interface):
                info = qtm_instance.info.decode("utf-8").split(",")[0]

                if not info in self._found_qtms:
                    self.discoveredQTM.emit(info, qtm_instance.host)
                    self._found_qtms[info] = True
        except Exception:
            pass

        self.discovering = False


class MainUI(QMainWindow, main_window_class):
    def __init__(self, *args):
        super().__init__(*args)
        self.setupUi(self)

        # Discovery
        self._discovery = QDiscovery()
        self._discovery.discoveringChanged.connect(self._is_discovering)
        self._discovery.discoveredQTM.connect(self._qtm_discovered)
        self.discover_button.clicked.connect(self._discovery.discover)

        # Connection
        self.connect_button.clicked.connect(self.connect_qtm)
        self.disconnect_button.clicked.connect(self.disconnect_qtm)
        self._is_streaming = False

        # # Settings
        # for setting in [
        #     "analog",
        #     "general",
        #     "3d",
        #     "6d",
        #     "all",
        #     "force",
        #     "gazevector",
        #     "image",
        # ]:
        #     self.settings_combo.addItem(setting)
        # self.settings_combo.currentIndexChanged[str].connect(
        #     self._settings_index_changed
        # )

        self._to_be_cleared = [
            self.settings_viewer,
        ]

        self._discovery.discover()

    def _is_discovering(self, discovering):
        if discovering:
            self.qtm_combo.clear()
        self.discover_button.setEnabled(not discovering)

    def _settings_index_changed(self, setting):
        start_async_task(self._get_settings(setting))

    def _qtm_discovered(self, info, ip):
        self.qtm_combo.addItem("{} {}".format(info, ip))
        self.connect_button.setEnabled(True)

    def connect_qtm(self):
        self.connect_button.setEnabled(False)
        self.discover_button.setEnabled(False)
        self.qtm_combo.setEnabled(False)

        start_async_task(self._connect_qtm())

    async def _connect_qtm(self):
        ip = self.qtm_combo.currentText().split(" ")[1]

        self._connection = await qtm.connect(
            ip, on_disconnect=self.on_disconnect, on_event=self.on_event, version=1.1
        )

        if self._connection is None:
            self.on_disconnect("Failed to connect")
            return

        await self._connection.take_control("gait1")
        await self._connection.get_state()

        self.disconnect_button.setEnabled(True)
        self.settings_combo.setEnabled(True)

    def disconnect_qtm(self):
        self._connection.disconnect()

    def on_disconnect(self, reason):
        self.disconnect_button.setEnabled(False)
        self.connect_button.setEnabled(True)
        self.discover_button.setEnabled(True)
        self.qtm_combo.setEnabled(True)
        self.settings_combo.setEnabled(False)

        for item in self._to_be_cleared:
            item.clear()
        self.settings_combo.clear()

    def on_packet(self, packet):
        if qtm.packet.QRTComponentType.ComponentAnalog in packet.components:
            _, analogs = packet.get_analog()
            if analogs:
                # print(type(analogs[0][2]))
                for child in analogs[0][2]: 
                    print("analogs output is : {}".format(child))


    def on_event(self, event):
        start_async_task(self._async_event_handler(event))

    async def _async_event_handler(self, event):

        if event == QRTEvent.EventRTfromFileStarted or event == QRTEvent.EventConnected:
            await self._setup_qtm()

        elif (
            event == QRTEvent.EventRTfromFileStopped
            or event == QRTEvent.EventConnectionClosed
        ) and self._is_streaming:
            start_async_task(self._stop_stream())

    async def _setup_qtm(self, stream=True):
        await self._get_analogs()
        # await self._get_settings(self.settings_combo.currentText())
        await self._start_stream()


    async def _get_settings(self, setting="analog"):
        result = await self._connection.get_parameters(parameters=[setting])

        self.settings_viewer.setText(result.decode("utf-8"))

    async def _get_analogs(self):
        result = await self._connection.get_parameters(parameters=["analog"])
        print('========================================================')
        print('List des analogs')
        try:
            xml = ET.fromstring(result)
            for label in (label.text for label in xml.iter("Label")):
                print(label)
                self.settings_combo.addItem(label)
        except ET.ParseError:
            print(result)
            return
        print('========================================================')


    async def _stop_stream(self):
        await self._connection.stream_frames_stop()
        self._is_streaming = False

    async def _start_stream(self):
        result = await self._connection.stream_frames(
            frames="frequency:10", components=["analog"], on_packet=self.on_packet
        )
        if result == b"Ok":
            self._is_streaming = True


def main():

    app = QApplication(sys.argv)

    # Create and set an event loop that combines qt and asyncio
    loop = QSelectorEventLoop(app)
    asyncio.set_event_loop(loop)

    main_window = MainUI()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
