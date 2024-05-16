def clearsubLayout(layout):
    if layout is not None:
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                clearsubLayout(item)
                layout.removeItem(item)
            else:
                raise Exception("something wrong in deleting layouts")